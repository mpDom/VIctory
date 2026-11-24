//! Host descriptor heaps behind the title's shader-visible heaps (decision
//! 0007, step 3). Translated shaders index the heaps directly at the
//! title's descriptor positions, so each host heap keeps the title heap's
//! order: slot `i` is made from what the title wrote at slot `i`, when it
//! writes it (view creation and descriptor copies) and again when a draw
//! reads it through a table and finds the slot changed.

use std::sync::Mutex;

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

use super::{VIEW_TAG, View, view_at};
use crate::host::{Host, host};

pub(crate) struct HostHeap {
    pub(crate) heap: ID3D12DescriptorHeap,
    cpu: usize,
    step: u32,
    /// Slots mirroring the title's; `spare` more follow them, and in a
    /// view heap its guard slot after those ([`HostHeap::guard_slot`]).
    count: u32,
    /// Dynamic slots past the title's: [`DYNAMIC`] in a view heap,
    /// [`SAMPLER_DYNAMIC`] in a capped sampler heap, none in another.
    spare: u32,
    sampler: bool,
    /// The record (view index and generation) each slot was last made from.
    made: Mutex<Vec<u64>>,
    /// The host resource each slot views, held while the slot does: shaders
    /// index heaps directly, so a slot the title has not rewritten since it
    /// released the resource may still be read, and a freed host resource
    /// read there is a GPU page fault (the console reads stale memory).
    kept: Mutex<Vec<Option<ID3D12Resource>>>,
    /// Where the search for free dynamic slots starts, and which are taken
    /// (by lists not yet reset).
    next: Mutex<(u32, Vec<bool>)>,
}

/// Host heaps' GPU ranges (start, bytes, bytes per slot, sampler), for
/// device-removed reports: a shader indexing a heap past its end faults.
static GPU_RANGES: Mutex<Vec<(u64, u64, u64, bool)>> = Mutex::new(Vec::new());

/// For device-removed reports: where host GPU address `va` lies against
/// the host descriptor heaps.
pub(crate) fn near_heaps(va: u64) -> String {
    let ranges = GPU_RANGES.lock().unwrap_or_else(|e| e.into_inner());
    let mut text = String::new();
    for &(start, bytes, step, sampler) in ranges.iter() {
        let kind = if sampler { "sampler" } else { "view" };
        let place = if va < start {
            format!("{:#x} below", start - va)
        } else if va < start + bytes {
            format!("inside, slot {}", (va - start) / step)
        } else {
            format!("{:#x} past its end (slot {})", va - start - bytes, (va - start) / step)
        };
        text.push_str(&format!("; {kind} heap {start:#x}+{bytes:#x}: {place}"));
    }
    text
}

/// The host's limit for shader-visible view heaps on resource binding
/// tiers 1 and 2; tier 3 hardware may take more.
const PORTABLE_VIEWS: u32 = 1_000_000;

/// Slots past the title's in a view heap, for tables a draw needs made
/// afresh (constant buffers copied per draw). A list holds the slots it
/// took until it is reset (its submissions have run) or released.
/// 2^16 ran out seconds into the world (every world run on an RX 9070 XT,
/// each a device removal shortly after; #59).
const DYNAMIC: u32 = 1 << 18;

/// The host's limit for shader-visible sampler heaps, on every binding
/// tier.
const SAMPLERS: u32 = D3D12_MAX_SHADER_VISIBLE_SAMPLER_HEAP_SIZE;

/// Dynamic slots of a sampler heap capped at [`SAMPLERS`], for tables
/// reaching past the title slots it keeps.
const SAMPLER_DYNAMIC: u32 = 256;

impl HostHeap {
    pub(crate) fn new(sampler: bool, count: u32) -> Option<Self> {
        let host = host()?;
        let kind = if sampler {
            D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER
        } else {
            D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV
        };
        let make = |count: u32| -> windows::core::Result<ID3D12DescriptorHeap> {
            // SAFETY: plain heap creation.
            unsafe {
                host.device
                    .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                        Type: kind,
                        NumDescriptors: count,
                        Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                        NodeMask: 0,
                    })
            }
        };
        if sampler && count > SAMPLERS {
            // shortcut: samplers past the host's limit are left out; tables
            // reaching them are made afresh in the dynamic slots.
            tracing::warn!(count, "host sampler heap capped at {SAMPLERS}");
            let heap = match make(SAMPLERS) {
                Ok(heap) => heap,
                Err(e) => {
                    tracing::warn!(error = %e, count, sampler, "no host descriptor heap");
                    return None;
                }
            };
            return Self::with(
                host,
                heap,
                kind,
                SAMPLERS - SAMPLER_DYNAMIC,
                SAMPLER_DYNAMIC,
            );
        }
        let extra = if sampler { 0 } else { DYNAMIC };
        // A view heap ends with its guard slot, after the dynamic ones (see
        // [`HostHeap::guard_slot`]).
        let guard = u32::from(!sampler);
        let (heap, count) = match make(count + extra + guard) {
            Ok(heap) => (heap, count),
            Err(e) if !sampler && count + extra + guard > PORTABLE_VIEWS => {
                // shortcut: descriptors past the host's limit are left out;
                // a title that reaches them needs remapping.
                tracing::warn!(error = %e, count, "host view heap capped at {PORTABLE_VIEWS}");
                (make(PORTABLE_VIEWS).ok()?, PORTABLE_VIEWS - extra - guard)
            }
            Err(e) => {
                tracing::warn!(error = %e, count, sampler, "no host descriptor heap");
                return None;
            }
        };
        Self::with(host, heap, kind, count, extra)
    }

    /// The host heap over `heap`: `count` title slots, then `extra` dynamic
    /// ones, then in a view heap the guard slot, all made null.
    fn with(
        host: &Host,
        heap: ID3D12DescriptorHeap,
        kind: D3D12_DESCRIPTOR_HEAP_TYPE,
        count: u32,
        extra: u32,
    ) -> Option<Self> {
        let sampler = kind == D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER;
        // SAFETY: a live heap.
        let (cpu, step) = unsafe {
            (
                heap.GetCPUDescriptorHandleForHeapStart().ptr,
                host.device.GetDescriptorHandleIncrementSize(kind),
            )
        };
        tracing::debug!(count, sampler, "host descriptor heap");
        // SAFETY: a live shader-visible heap.
        let gpu = unsafe { heap.GetGPUDescriptorHandleForHeapStart().ptr };
        let h = HostHeap {
            heap,
            cpu,
            step,
            count,
            spare: extra,
            sampler,
            made: Mutex::new(vec![0; count as usize]),
            kept: Mutex::new(vec![None; (count + extra) as usize]),
            next: Mutex::new((0, vec![false; extra as usize])),
        };
        // A slot never written is undefined: shaders reading one (a slot
        // the title wrote in a way we do not know, or a table entry no
        // used-resource entry names) read a wild address. Every slot starts
        // null instead.
        let started = std::time::Instant::now();
        // A view heap's guard slot follows its dynamic slots, outside the
        // pool `dynamic` hands out.
        let total = count + extra + u32::from(!sampler);
        if let Err(e) = h.fill_null(kind, total) {
            tracing::warn!(error = %e, "host descriptor heap not made null");
            return None;
        }
        GPU_RANGES.lock().unwrap_or_else(|e| e.into_inner()).push((
            gpu,
            u64::from(total) * u64::from(step),
            u64::from(step),
            sampler,
        ));
        tracing::debug!(ms = started.elapsed().as_millis(), "host descriptor heap made null");
        Some(h)
    }

    /// Makes the first `total` slots null: one null slot in a CPU-side heap,
    /// doubled by copies to a block, the block copied over the heap.
    fn fill_null(&self, kind: D3D12_DESCRIPTOR_HEAP_TYPE, total: u32) -> windows::core::Result<()> {
        let Some(host) = host() else { return Ok(()) };
        let block_count = total.min(if self.sampler {
            D3D12_MAX_SHADER_VISIBLE_SAMPLER_HEAP_SIZE
        } else {
            1 << 16
        });
        // SAFETY: plain heap creation and copies inside live heaps.
        unsafe {
            let block: ID3D12DescriptorHeap =
                host.device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: kind,
                    NumDescriptors: block_count,
                    ..Default::default()
                })?;
            let start = block.GetCPUDescriptorHandleForHeapStart();
            let at = |i: u32| D3D12_CPU_DESCRIPTOR_HANDLE {
                ptr: start.ptr + (i * self.step) as usize,
            };
            if self.sampler {
                null_sampler(start);
            } else {
                null_view(start);
            }
            let mut filled = 1;
            while filled < block_count {
                let n = filled.min(block_count - filled);
                host.device.CopyDescriptorsSimple(n, at(filled), start, kind);
                filled += n;
            }
            let mut done = 0;
            while done < total {
                let n = block_count.min(total - done);
                host.device.CopyDescriptorsSimple(n, self.cpu(done), start, kind);
                done += n;
            }
        }
        Ok(())
    }

    /// A plain point sampler at slot `index`, for sampler slots without a
    /// view.
    fn null_sampler(&self, index: u32) {
        null_sampler(self.cpu(index));
    }
}

/// A plain point sampler at `at`.
fn null_sampler(at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    let Some(host) = host() else { return };
    let desc = D3D12_SAMPLER_DESC {
        Filter: D3D12_FILTER_MIN_MAG_MIP_POINT,
        AddressU: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        AddressV: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        AddressW: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        MaxAnisotropy: 1,
        ComparisonFunc: D3D12_COMPARISON_FUNC_NEVER,
        ..Default::default()
    };
    // SAFETY: a slot of a sampler heap of ours.
    unsafe { host.device.CreateSampler(&desc, at) };
}

impl HostHeap {
    /// `n` consecutive free dynamic slots (view heaps and capped sampler
    /// heaps), taken until [`HostHeap::release`]: the first's index.
    pub(crate) fn dynamic(&self, n: u32) -> Option<u32> {
        if n == 0 || self.spare == 0 {
            return None;
        }
        if n > self.spare {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| {
                tracing::warn!(
                    n,
                    spare = self.spare,
                    sampler = self.sampler,
                    "a table larger than the dynamic slots is not made"
                );
            });
            return None;
        }
        let spare = self.spare as usize;
        let mut guard = self.next.lock().unwrap_or_else(|e| e.into_inner());
        let (next, taken) = &mut *guard;
        let n = n as usize;
        let Some(first) = free_table(taken, *next as usize, n) else {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| {
                let (free, longest) = free_space(taken);
                tracing::warn!(
                    n,
                    cap = taken.len(),
                    free,
                    longest,
                    "no free dynamic descriptor slots: a table is not made"
                );
            });
            return None;
        };
        taken[first..first + n].fill(true);
        *next = ((first + n) % spare) as u32;
        Some(self.count + first as u32)
    }

    /// The title slots this heap has host slots for; fewer than the title's
    /// when the host caps its size.
    pub(crate) fn count(&self) -> u32 {
        self.count
    }

    /// The slot translated shaders read in place of a computed heap index
    /// at or past it (a stale buffer can give any index, and one past the
    /// heap hangs some GPUs): in a view heap its last slot, after the
    /// dynamic ones, null from the start and never made; in a sampler heap
    /// its last slot (any sampler reads safely), which in a capped sampler
    /// heap is its last dynamic slot, so that tables made in the dynamic
    /// slots stay below it.
    pub(crate) fn guard_slot(&self) -> u32 {
        (self.count + self.spare).saturating_sub(u32::from(self.sampler))
    }

    /// Gives back `n` dynamic slots from `first` (an index [`dynamic`]
    /// returned).
    ///
    /// [`dynamic`]: HostHeap::dynamic
    pub(crate) fn release(&self, first: u32, n: u32) {
        let Some(at) = first.checked_sub(self.count) else {
            return;
        };
        let mut guard = self.next.lock().unwrap_or_else(|e| e.into_inner());
        let end = at.saturating_add(n).min(self.spare) as usize;
        if let Some(slots) = guard.1.get_mut(at as usize..end) {
            slots.fill(false);
        }
    }

    /// Makes slot `index` (a dynamic one) from `view`; a null view if the
    /// host cannot.
    pub(crate) fn make_at(&self, index: u32, view: &View) {
        let kept = make(view, self.cpu(index));
        if kept.is_none() {
            if self.sampler {
                self.null_sampler(index);
            } else {
                null_for(view, self.cpu(index));
            }
        }
        self.keep(index, kept.flatten());
    }

    /// Makes slot `index` read nothing: a null view, or a plain sampler.
    pub(crate) fn make_null(&self, index: u32) {
        if self.sampler {
            self.null_sampler(index);
        } else {
            null_view(self.cpu(index));
        }
        self.keep(index, None);
    }

    /// Slot `index` now views `resource` (or nothing of the host's).
    fn keep(&self, index: u32, resource: Option<ID3D12Resource>) {
        let mut kept = self.kept.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(slot) = kept.get_mut(index as usize) {
            *slot = resource;
        }
    }

    /// Makes slot `index` (a dynamic one) a constant buffer view of host
    /// memory (the list's own, which it holds).
    pub(crate) fn constant_at(&self, index: u32, gpu: u64, size: u32) {
        self.keep(index, None);
        let Some(host) = host() else { return };
        let desc = D3D12_CONSTANT_BUFFER_VIEW_DESC {
            BufferLocation: gpu,
            SizeInBytes: size.next_multiple_of(256),
        };
        // SAFETY: a slot of our heap; the memory is the caller's.
        unsafe {
            host.device
                .CreateConstantBufferView(Some(&desc), self.cpu(index))
        };
    }

    /// Makes slot `index` (a dynamic one) a raw buffer view of `len` bytes
    /// of `buffer` from byte `offset` (a multiple of 16): a root SRV with a
    /// size, whose reads past it return zeros instead of reaching whatever
    /// the host placed after the buffer.
    pub(crate) fn raw_at(&self, index: u32, buffer: &ID3D12Resource, offset: u64, len: u64) {
        let Some(host) = host() else { return };
        let elements = (len / 4).clamp(1, u64::from(u32::MAX)) as u32;
        let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: DXGI_FORMAT_R32_TYPELESS,
            ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Buffer: D3D12_BUFFER_SRV {
                    FirstElement: offset / 4,
                    NumElements: elements,
                    StructureByteStride: 0,
                    Flags: D3D12_BUFFER_SRV_FLAG_RAW,
                },
            },
        };
        // SAFETY: a slot of our heap and a live buffer covering the range.
        unsafe {
            host.device
                .CreateShaderResourceView(buffer, Some(&desc), self.cpu(index))
        };
        self.keep(index, Some(buffer.clone()));
    }

    /// Makes slot `index` a raw buffer view that reads zeros.
    pub(crate) fn null_raw_at(&self, index: u32) {
        null_buffer_view(self.cpu(index));
        self.keep(index, None);
    }

    /// Whether this is a sampler heap.
    pub(crate) fn is_sampler(&self) -> bool {
        self.sampler
    }

    /// The shader-visible handle of slot `index`.
    pub(crate) fn gpu_handle(&self, index: u32) -> D3D12_GPU_DESCRIPTOR_HANDLE {
        // SAFETY: a live shader-visible heap.
        let start = unsafe { self.heap.GetGPUDescriptorHandleForHeapStart().ptr };
        D3D12_GPU_DESCRIPTOR_HANDLE {
            ptr: start + u64::from(index) * u64::from(self.step),
        }
    }

    fn cpu(&self, index: u32) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        D3D12_CPU_DESCRIPTOR_HANDLE {
            ptr: self.cpu + index as usize * self.step as usize,
        }
    }

    /// Makes host slot `index` from the record the title has at `handle`
    /// if it changed since, and returns the view recorded there. Buffer
    /// views are made only for draws (`for_draw`), which also copy their
    /// data: a host copy of a buffer is made on first use.
    pub(crate) fn mirror(&self, index: u32, handle: u64, for_draw: bool) -> Option<View> {
        if index >= self.count {
            // Diagnostic (NVIDIA heap cap): the highest title slot past the
            // host heap seen so far.
            static HIGHEST: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            if HIGHEST.fetch_max(index, std::sync::atomic::Ordering::Relaxed) < index {
                tracing::warn!(
                    index,
                    count = self.count,
                    for_draw,
                    "title slot past the host heap"
                );
            }
            return None;
        }
        // SAFETY: the handle is a descriptor slot in title memory.
        let [tag, i, generation] = unsafe { (handle as *const [u32; 3]).read_unaligned() };
        // Generations start at 1, so a made slot's key is never 0.
        let key = u64::from(generation) << 32 | u64::from(i);
        let mut made = self.made.lock().unwrap_or_else(|e| e.into_inner());
        let view = (tag == VIEW_TAG).then(|| view_at(i, generation)).flatten();
        let Some(view) = view else {
            // Not a view of ours (never written, or of a resource the
            // title released): nothing to read.
            if made[index as usize] != 0 {
                made[index as usize] = 0;
                self.make_null(index);
            }
            return None;
        };
        if made[index as usize] != key {
            let later = !for_draw && view.is_buffer();
            made[index as usize] = if later { 0 } else { key };
            let kept = if later {
                None
            } else {
                make(&view, self.cpu(index))
            };
            if kept.is_none() && !self.sampler {
                null_for(&view, self.cpu(index));
            }
            self.keep(index, kept.flatten());
        }
        Some(view)
    }

    /// The view host slot `index` was made from, in a few words, with the
    /// title and host addresses of a buffer view (diagnostics). "null" for
    /// a slot never made or left null.
    #[cfg(feature = "diagnostics")]
    pub(crate) fn describe_slot(&self, index: u32) -> Option<String> {
        let key = *self.made.lock().unwrap_or_else(|e| e.into_inner()).get(index as usize)?;
        if key == 0 {
            return Some("null".into());
        }
        let view = view_at(key as u32, (key >> 32) as u32)?;
        let mut text: String = format!("{view:?}").chars().take(120).collect();
        if let Some((address, len)) = buffer_range(&view) {
            let host = crate::shadow::find(address, len.max(1)).map(|s| s.gpu(address));
            text.push_str(&format!(" title {address:#x}+{len:#x} host {host:#x?}"));
        }
        Some(text)
    }
}

/// Total free slots and the longest contiguous run in the physical heap.
fn free_space(taken: &[bool]) -> (usize, usize) {
    let mut free = 0;
    let mut longest = 0;
    for gap in taken.split(|&held| held) {
        free += gap.len();
        longest = longest.max(gap.len());
    }
    (free, longest)
}

/// Try each possible table start from the cursor, then from slot zero.
/// A table may cross the cursor on the second pass, but must fit in the heap.
fn free_table(taken: &[bool], cursor: usize, n: usize) -> Option<usize> {
    let end = taken.len().checked_sub(n)? + 1;
    let split = cursor.min(end);
    for starts in [split..end, 0..split] {
        let mut first = starts.start;
        while first < starts.end {
            match taken[first..first + n].iter().position(|&held| held) {
                Some(offset) => first += offset + 1,
                None => return Some(first),
            }
        }
    }
    None
}

/// Logs `what` the first time it happens.
macro_rules! once {
    ($($arg:tt)*) => {{
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| tracing::warn!($($arg)*));
    }};
}

/// A view description from its words, with the format as the host has it.
fn srv_desc(words: &[u32; 10]) -> D3D12_SHADER_RESOURCE_VIEW_DESC {
    let mut words = *words;
    words[0] = crate::resource::pc_format(words[0]);
    // SAFETY: 40 bytes in the host's layout.
    unsafe {
        std::ptr::from_ref(&words)
            .cast::<D3D12_SHADER_RESOURCE_VIEW_DESC>()
            .read_unaligned()
    }
}

fn uav_desc(words: &[u32; 10]) -> D3D12_UNORDERED_ACCESS_VIEW_DESC {
    let mut words = *words;
    words[0] = crate::resource::pc_format(words[0]);
    // SAFETY: 40 bytes in the host's layout.
    unsafe {
        std::ptr::from_ref(&words)
            .cast::<D3D12_UNORDERED_ACCESS_VIEW_DESC>()
            .read_unaligned()
    }
}

/// A buffer view of title memory at `address` on its host copy: the first
/// element moves by the copy's offset.
/// Bytes per element of a buffer view.
fn element_size(desc: &D3D12_SHADER_RESOURCE_VIEW_DESC) -> u64 {
    // SAFETY: a buffer view's union member.
    let buffer = unsafe { desc.Anonymous.Buffer };
    if buffer.StructureByteStride > 0 {
        u64::from(buffer.StructureByteStride)
    } else if buffer.Flags.0 & D3D12_BUFFER_SRV_FLAG_RAW.0 != 0 {
        4
    } else {
        crate::resource::element(desc.Format.0 as u32).map_or(0, |(bytes, _)| bytes)
    }
}

/// The title memory a buffer shader or unordered access view reads: its
/// address and length.
pub(crate) fn buffer_range(view: &View) -> Option<(u64, u64)> {
    let (base, words): (u64, &[u32; 10]) = match view {
        View::Shader(r, Some(desc)) => (
            crate::resource::buffer_bytes(*r as crate::object::This)?.0 as u64,
            desc,
        ),
        View::Placed(false, resource, desc) if resource[0] == 1 => (
            u64::from(desc[10]) | u64::from(desc[11]) << 32,
            desc[..10].try_into().ok()?,
        ),
        View::Unordered(r, _, Some(desc)) => {
            let base = crate::resource::buffer_bytes(*r as crate::object::This)?.0 as u64;
            return unordered_range(base, desc);
        }
        View::Placed(true, resource, desc) if resource[0] == 1 => {
            let base = u64::from(desc[10]) | u64::from(desc[11]) << 32;
            return unordered_range(base, desc[..10].try_into().ok()?);
        }
        _ => return None,
    };
    let desc = srv_desc(words);
    let element = element_size(&desc);
    // SAFETY: a buffer view's union member.
    let buffer = unsafe { desc.Anonymous.Buffer };
    Some((
        base + buffer.FirstElement * element,
        u64::from(buffer.NumElements) * element,
    ))
}

/// [`buffer_range`] of an unordered access view of memory at `base`.
fn unordered_range(base: u64, words: &[u32; 10]) -> Option<(u64, u64)> {
    let desc = uav_desc(words);
    // SAFETY: a buffer view's union member.
    let buffer = unsafe { desc.Anonymous.Buffer };
    let element = if buffer.StructureByteStride > 0 {
        u64::from(buffer.StructureByteStride)
    } else if buffer.Flags.0 & D3D12_BUFFER_UAV_FLAG_RAW.0 != 0 {
        4
    } else {
        crate::resource::element(desc.Format.0 as u32)?.0
    };
    Some((
        base + buffer.FirstElement * element,
        u64::from(buffer.NumElements) * element,
    ))
}

/// Whether a raw buffer view may start at element `first` (4 bytes each):
/// the host wants 16-byte aligned starts, the console only 4, and refuses
/// the call (removing the device) for the rest.
/// shortcut: a view the host refuses reads nothing (a null view); starting
/// it 16 bytes lower would need the shader to add the difference.
fn raw_start_fits(first: u64) -> bool {
    first.is_multiple_of(4)
}

/// The host resource a slot made by [`make`] views, if any.
type Kept = Option<ID3D12Resource>;

/// A structured view's bytes as a raw view of its host copy: the first
/// dword and the dword count, for a view `offset` bytes into the copy whose
/// description starts at element `first` and has `count` elements of
/// `stride` bytes.
///
/// The host places structured element `i` at byte `i * stride` of the
/// resource, so a structured view whose start in the copy is not a multiple
/// of its stride (a title buffer's view 16 bytes past a 32-byte element)
/// cannot be made there. A raw view of the same bytes can if its start is
/// 16-byte aligned (`D3D12_RAW_UAV_SRV_BYTE_ALIGNMENT`; the host removes the
/// device for one that is not), and it reads and writes the copy every other
/// view of that memory uses. Translated shaders read buffer views raw unless
/// they index them (`victory-shader`, "Buffer views in the heap").
fn raw_window(offset: u64, first: u64, count: u32, stride: u32) -> Option<(u64, u32)> {
    if stride == 0 || !stride.is_multiple_of(4) {
        return None;
    }
    let start = offset.checked_add(first.checked_mul(u64::from(stride))?)?;
    if !start.is_multiple_of(u64::from(D3D12_RAW_UAV_SRV_BYTE_ALIGNMENT)) {
        return None;
    }
    let dwords = u64::from(count) * u64::from(stride / 4);
    Some((start / 4, u32::try_from(dwords).ok()?))
}

fn buffer_srv(
    address: u64,
    len: u64,
    mut desc: D3D12_SHADER_RESOURCE_VIEW_DESC,
    at: D3D12_CPU_DESCRIPTOR_HANDLE,
) -> Option<Kept> {
    let shadow = crate::shadow::find(address, len)?;
    let element = element_size(&desc);
    // SAFETY: a buffer view's union member.
    let buffer = unsafe { &mut desc.Anonymous.Buffer };
    let offset = shadow.offset(address);
    if element != 0 && offset % element == 0 {
        buffer.FirstElement += offset / element;
    } else if let Some((first, count)) = raw_window(
        offset,
        buffer.FirstElement,
        buffer.NumElements,
        buffer.StructureByteStride,
    ) {
        // shortcut: a shader that indexes the view declares it structured,
        // and D3D12 leaves structured reads of a raw view undefined (NVIDIA
        // and WARP read the right bytes).
        once!(
            offset,
            element,
            "structured buffer view off its host copy's elements made raw"
        );
        *buffer = D3D12_BUFFER_SRV {
            FirstElement: first,
            NumElements: count,
            StructureByteStride: 0,
            Flags: D3D12_BUFFER_SRV_FLAG_RAW,
        };
        desc.Format = DXGI_FORMAT_R32_TYPELESS;
    } else {
        once!(
            offset,
            element,
            "buffer view not on an element of its host copy"
        );
        return None;
    }
    if buffer.Flags.0 & D3D12_BUFFER_SRV_FLAG_RAW.0 != 0 && !raw_start_fits(buffer.FirstElement) {
        let first = buffer.FirstElement;
        once!(first, "raw buffer view not on 16 bytes of its host copy");
        return None;
    }
    let host = host()?;
    // SAFETY: a live buffer and a slot of our heap.
    unsafe {
        host.device
            .CreateShaderResourceView(&shadow.buffer, Some(&desc), at)
    };
    Some(Some(shadow.buffer.clone()))
}

/// An unordered access view of title buffer memory at `address` (`len`
/// bytes) on its host copy, as [`buffer_srv`]; false if the copy cannot be
/// written by shaders (upload memory).
/// shortcut: what shaders write stays in the host copy; title memory (and
/// so the CPU, and copies run on the CPU) does not see it.
fn buffer_uav(
    address: u64,
    len: u64,
    mut desc: D3D12_UNORDERED_ACCESS_VIEW_DESC,
    at: D3D12_CPU_DESCRIPTOR_HANDLE,
) -> Option<Kept> {
    let shadow = crate::shadow::find(address, len)?;
    // SAFETY: a live buffer.
    let flags = unsafe { shadow.buffer.GetDesc() }.Flags;
    if flags.0 & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0 == 0 {
        once!("unordered access to a buffer whose host copy shaders cannot write");
        return None;
    }
    // SAFETY: a buffer view's union member.
    let buffer = unsafe { &mut desc.Anonymous.Buffer };
    let element = if buffer.StructureByteStride > 0 {
        u64::from(buffer.StructureByteStride)
    } else if buffer.Flags.0 & D3D12_BUFFER_UAV_FLAG_RAW.0 != 0 {
        4
    } else {
        crate::resource::element(desc.Format.0 as u32).map_or(0, |(bytes, _)| bytes)
    };
    let offset = shadow.offset(address);
    buffer.CounterOffsetInBytes = 0;
    if element != 0 && offset % element == 0 {
        buffer.FirstElement += offset / element;
    } else if let Some((first, count)) = raw_window(
        offset,
        buffer.FirstElement,
        buffer.NumElements,
        buffer.StructureByteStride,
    ) {
        // As in `buffer_srv`; what shaders write lands in the shared copy,
        // in the range `buffer_range` gives (copied back to title memory).
        once!(
            offset,
            element,
            "structured unordered buffer view off its host copy's elements made raw"
        );
        *buffer = D3D12_BUFFER_UAV {
            FirstElement: first,
            NumElements: count,
            StructureByteStride: 0,
            CounterOffsetInBytes: 0,
            Flags: D3D12_BUFFER_UAV_FLAG_RAW,
        };
        desc.Format = DXGI_FORMAT_R32_TYPELESS;
    } else {
        once!(offset, element, "unordered buffer view not on an element of its host copy");
        return None;
    }
    if buffer.Flags.0 & D3D12_BUFFER_UAV_FLAG_RAW.0 != 0 && !raw_start_fits(buffer.FirstElement) {
        let first = buffer.FirstElement;
        once!(
            first,
            "raw unordered buffer view not on 16 bytes of its host copy"
        );
        return None;
    }
    let host = host()?;
    // SAFETY: a live buffer and a slot of our heap.
    unsafe {
        host.device
            .CreateUnorderedAccessView(&shadow.buffer, None::<&ID3D12Resource>, Some(&desc), at)
    };
    Some(Some(shadow.buffer.clone()))
}

/// Whether the host lets views of a typed resource name another format of
/// its family (`CastingFullyTypedFormatSupported`; never SNORM for UNORM,
/// nor FLOAT for another type), which [`fits`] uses for sRGB and linear
/// twins only.
fn casts_typed() -> bool {
    static CASTS: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CASTS.get_or_init(|| {
        let Some(host) = host() else { return false };
        let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS3::default();
        // SAFETY: the query writes its own structure of the given size.
        let asked = unsafe {
            host.device.CheckFeatureSupport(
                D3D12_FEATURE_D3D12_OPTIONS3,
                (&raw mut options).cast(),
                size_of_val(&options) as u32,
            )
        };
        let casts = asked.is_ok() && options.CastingFullyTypedFormatSupported.as_bool();
        if !casts {
            tracing::warn!(
                "the host does not cast typed formats: sRGB views of linear textures (and the reverse) read in the texture's own format"
            );
        }
        casts
    })
}

/// Adapts the requested view format to a format compatible with host D3D12.
pub(crate) fn adapt_view_format(
    own: u32,
    mut format: Option<&mut DXGI_FORMAT>,
    unordered: bool,
) -> bool {
    use crate::resource::{family, first_concrete, is_typeless};
    let typeless = is_typeless(own);
    if let Some(format) = format.as_deref_mut()
        && unordered
        && let Some(&(_, linear)) = crate::resource::SRGB
            .iter()
            .find(|&&(s, _)| s == format.0 as u32)
    {
        format.0 = linear as i32;
    }
    let view = format.as_ref().map_or(0, |f| f.0 as u32);
    if view == 0 && format.is_none() {
        return !typeless;
    }
    if view != 0 && view != own && crate::resource::target_format(view) == own {
        if let Some(format) = format {
            *format = DXGI_FORMAT(own as i32);
        }
        return true;
    }
    if view != 0 && family(view) != family(own) {
        once!(view, own, "view format of another family");
        return false;
    }
    let twin = crate::resource::SRGB
        .iter()
        .any(|&(s, l)| (s, l) == (view, own) || (l, s) == (view, own));
    if let Some(format) = format {
        if !typeless && !(twin && casts_typed()) {
            *format = DXGI_FORMAT(own as i32);
        } else if typeless && (view == 0 || is_typeless(view)) {
            let concrete = first_concrete(own);
            *format = DXGI_FORMAT(concrete as i32);
        }
    }
    true
}

/// Whether the host can make a shader (or unordered access) view of
/// `texture` in `format` (none: the texture's own). The console allows any
/// view; the host refuses (and removes the device for) views without the
/// resource's permission, of typed depth, and in another format than a
/// typed resource's, but for casts it allows (see [`casts_typed`]).
fn fits(texture: &ID3D12Resource, format: Option<&mut DXGI_FORMAT>, unordered: bool) -> bool {
    use crate::resource::DEPTH_FORMATS;
    // SAFETY: a live resource.
    let d = unsafe { texture.GetDesc() };
    let allowed = if unordered {
        d.Flags.0 & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0 != 0
    } else {
        d.Flags.0 & D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE.0 == 0
    };
    let own = d.Format.0 as u32;
    if !allowed || DEPTH_FORMATS.iter().any(|&(typed, _)| typed == own) {
        once!(
            unordered,
            format = own,
            flags = d.Flags.0,
            "view the host does not allow"
        );
        return false;
    }
    adapt_view_format(own, format, unordered)
}

/// The unordered access view a null description makes of a 2D texture
/// (twins are): mip 0, every array slice; the format is the caller's.
fn whole_uav(texture: &ID3D12Resource) -> D3D12_UNORDERED_ACCESS_VIEW_DESC {
    // SAFETY: a live resource.
    let slices = u32::from(unsafe { texture.GetDesc() }.DepthOrArraySize);
    let mut desc = D3D12_UNORDERED_ACCESS_VIEW_DESC {
        ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2D,
        ..Default::default()
    };
    if slices > 1 {
        desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2DARRAY;
        desc.Anonymous.Texture2DArray = D3D12_TEX2D_ARRAY_UAV {
            ArraySize: slices,
            ..Default::default()
        };
    }
    desc
}

/// Makes the host descriptor for `view` at `at`: the host resource it
/// views, or `None` if it cannot.
fn make(view: &View, at: D3D12_CPU_DESCRIPTOR_HANDLE) -> Option<Kept> {
    let host = host()?;
    let device = &host.device;
    match view {
        &View::Constant(address, size) => {
            let shadow = crate::shadow::find(address, u64::from(size))?;
            let gpu = shadow.gpu(address);
            if gpu % 256 != 0 {
                once!(
                    address = format_args!("{address:#x}"),
                    "constant buffer not 256-byte aligned on the host"
                );
                return None;
            }
            let desc = D3D12_CONSTANT_BUFFER_VIEW_DESC {
                BufferLocation: gpu,
                SizeInBytes: size.next_multiple_of(256),
            };
            // SAFETY: a slot of our heap; the range lies in the host copy.
            unsafe { device.CreateConstantBufferView(Some(&desc), at) };
            Some(Some(shadow.buffer.clone()))
        }
        View::Shader(resource, desc) => {
            let resource = *resource as crate::object::This;
            if let Some(texture) = crate::resource::host_resource(resource) {
                let mut desc = desc.as_ref().map(srv_desc);
                if !fits(texture, desc.as_mut().map(|d| &mut d.Format), false) {
                    return None;
                }
                // SAFETY: a live texture and a slot of our heap.
                unsafe {
                    device.CreateShaderResourceView(
                        texture,
                        desc.as_ref().map(std::ptr::from_ref),
                        at,
                    )
                };
                Some(Some(texture.clone()))
            } else if let (Some((data, len)), Some(desc)) =
                (crate::resource::buffer_bytes(resource), desc)
            {
                buffer_srv(data as u64, len, srv_desc(desc), at)
            } else {
                None
            }
        }
        View::Unordered(resource, _counter, desc) => {
            let resource = *resource as crate::object::This;
            let Some((texture, twin_format)) = crate::resource::host_unordered(resource) else {
                return match (crate::resource::buffer_bytes(resource), desc) {
                    (Some((data, len)), Some(desc)) => {
                        buffer_uav(data as u64, len, uav_desc(desc), at)
                    }
                    _ => None,
                };
            };
            let mut desc = desc.as_ref().map(uav_desc);
            // A depth target's twin is typeless: views name a format of its
            // family; a depth one, the typeless one or none becomes the one
            // with the depth's bits.
            if let Some(format) = twin_format {
                use crate::resource::{DEPTH_FORMATS, family};
                let d = desc.get_or_insert_with(|| whole_uav(texture));
                let own = d.Format.0 as u32;
                if own == 0 || own == family(format) || DEPTH_FORMATS.iter().any(|&(f, _)| f == own)
                {
                    d.Format.0 = format as i32;
                }
            }
            if !fits(texture, desc.as_mut().map(|d| &mut d.Format), true) {
                return None;
            }
            // SAFETY: a live texture and a slot of our heap.
            unsafe {
                device.CreateUnorderedAccessView(
                    texture,
                    None::<&ID3D12Resource>,
                    desc.as_ref().map(std::ptr::from_ref),
                    at,
                )
            };
            Some(Some(texture.clone()))
        }
        View::Sampler(words) => {
            // SAFETY: 13 words in the host's layout.
            let desc = unsafe {
                std::ptr::from_ref(words)
                    .cast::<D3D12_SAMPLER_DESC>()
                    .read_unaligned()
            };
            // SAFETY: a slot of our sampler heap.
            unsafe { device.CreateSampler(&desc, at) };
            Some(None)
        }
        View::ConsoleSampler(words) => {
            let desc = crate::sampler::console(words);
            // SAFETY: a slot of our sampler heap.
            unsafe { device.CreateSampler(&desc, at) };
            Some(None)
        }
        View::Placed(unordered, resource, desc) => {
            if let Some(view) = placed_texture(view) {
                return make(&view, at);
            }
            const BUFFER: u32 = 1;
            if resource[0] != BUFFER {
                // shortcut: unordered access to a texture the title placed
                // itself is not mirrored; give it a resource as shader
                // views of such textures get when a dispatch writes one.
                once!(dimension = resource[0], "placed unordered texture view not mirrored");
                return None;
            }
            // The resource description's width, after the alignment.
            let len = u64::from(resource[4]) | u64::from(resource[5]) << 32;
            let address = u64::from(desc[10]) | u64::from(desc[11]) << 32;
            let words: &[u32; 10] = desc[..10].try_into().unwrap_or(&[0; 10]);
            if *unordered {
                buffer_uav(address, len, uav_desc(words), at)
            } else {
                buffer_srv(address, len, srv_desc(words), at)
            }
        }
    }
}

/// A shader view of a texture the title placed in memory itself, as a view
/// of the resource made for that memory (on first use).
pub(crate) fn placed_texture(view: &View) -> Option<View> {
    const BUFFER: u32 = 1;
    let View::Placed(false, resource, desc) = view else {
        return None;
    };
    if resource[0] == BUFFER {
        return None;
    }
    let address = u64::from(desc[10]) | u64::from(desc[11]) << 32;
    let texture = crate::resource::texture_at(address, resource)?;
    Some(View::Shader(texture as usize, desc[..10].try_into().ok()))
}

/// A null view of the kind `view` is, for a slot whose view the host
/// cannot make: unordered access (to a buffer or a 2D texture) or a shader
/// view. Writes to it are dropped, reads give zeros.
fn null_for(view: &View, at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    let Some(host) = host() else { return };
    let is_buffer = |r: &usize| crate::resource::buffer_bytes(*r as crate::object::This).is_some();
    // Unordered views: whether a buffer, and the title's view dimension.
    let (buffer, dimension) = match view {
        View::Unordered(r, _, desc) => (
            is_buffer(r) || desc.is_some_and(|d| d[1] == 1),
            desc.map(|d| d[1]),
        ),
        View::Placed(true, resource, desc) => (resource[0] == 1, Some(desc[1])),
        // Shader views of buffers read nothing through a null buffer view:
        // a null 2D view where the shader expects a buffer is a mismatch
        // the hardware does not survive in general (GPU-based validation
        // reports it).
        View::Shader(r, desc) if is_buffer(r) || desc.is_some_and(|d| d[1] == 1) => {
            return null_buffer_view(at);
        }
        View::Placed(false, resource, _) if resource[0] == 1 => return null_buffer_view(at),
        // Textures: a null view of the title's dimension (1D, arrays, 3D,
        // cubes), not always 2D.
        View::Shader(_, Some(desc)) => return null_like(desc, at),
        View::Placed(false, _, desc) => {
            return null_like(desc[..10].try_into().unwrap_or(&[0; 10]), at);
        }
        View::Constant(..) => {
            let desc = D3D12_CONSTANT_BUFFER_VIEW_DESC::default();
            // SAFETY: a slot of our heap; a zero location makes a null view.
            return unsafe { host.device.CreateConstantBufferView(Some(&desc), at) };
        }
        _ => return null_view(at),
    };
    let desc = if buffer {
        D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: DXGI_FORMAT_R32_UINT,
            ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
            ..Default::default()
        }
    } else {
        null_uav_desc(dimension)
    };
    // SAFETY: a slot of our heap; a null resource is allowed with a desc.
    unsafe {
        host.device.CreateUnorderedAccessView(
            None::<&ID3D12Resource>,
            None::<&ID3D12Resource>,
            Some(&desc),
            at,
        )
    };
}

/// A null texture UAV of the title's view dimension (1D, 1D or 2D arrays,
/// 3D), 2D if it names none: a 2D null view where the shader writes a 3D
/// texture is a mismatch GPU-based validation reports (the world's wind
/// field, CS_CopyWindFieldBeforeUpdateWithSmooth, 2026-11-24).
fn null_uav_desc(dimension: Option<u32>) -> D3D12_UNORDERED_ACCESS_VIEW_DESC {
    let mut desc = D3D12_UNORDERED_ACCESS_VIEW_DESC {
        Format: DXGI_FORMAT_R32_UINT,
        ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2D,
        ..Default::default()
    };
    match dimension.map(|d| D3D12_UAV_DIMENSION(d as i32)) {
        Some(D3D12_UAV_DIMENSION_TEXTURE1D) => {
            desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE1D;
        }
        Some(D3D12_UAV_DIMENSION_TEXTURE1DARRAY) => {
            desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE1DARRAY;
            desc.Anonymous.Texture1DArray = D3D12_TEX1D_ARRAY_UAV {
                ArraySize: 1,
                ..Default::default()
            };
        }
        Some(D3D12_UAV_DIMENSION_TEXTURE2DARRAY) => {
            desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2DARRAY;
            desc.Anonymous.Texture2DArray = D3D12_TEX2D_ARRAY_UAV {
                ArraySize: 1,
                ..Default::default()
            };
        }
        Some(D3D12_UAV_DIMENSION_TEXTURE3D) => {
            desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE3D;
            desc.Anonymous.Texture3D = D3D12_TEX3D_UAV {
                WSize: u32::MAX,
                ..Default::default()
            };
        }
        _ => {}
    }
    desc
}

/// A buffer view that reads zeros (raw, as translated shaders mostly read
/// buffers).
fn null_buffer_view(at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    let Some(host) = host() else { return };
    let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
        Format: DXGI_FORMAT_R32_TYPELESS,
        ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
        Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
            Buffer: D3D12_BUFFER_SRV {
                NumElements: 1,
                Flags: D3D12_BUFFER_SRV_FLAG_RAW,
                ..Default::default()
            },
        },
    };
    // SAFETY: a slot of our heap; a null resource is allowed with a desc.
    unsafe {
        host.device
            .CreateShaderResourceView(None::<&ID3D12Resource>, Some(&desc), at)
    };
}

/// A texture view that reads zeros with the dimension (and mip and array
/// counts) of the title's description `words`; 2D if it names none.
fn null_like(words: &[u32; 10], at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    const TEXTURES: std::ops::RangeInclusive<u32> = 2..=10;
    if !TEXTURES.contains(&words[1]) {
        return null_view(at);
    }
    let Some(host) = host() else { return };
    // One mip, one slice or cube, plane 0: fields a null view still checks.
    let mut w = [0u32; 10];
    w[1] = words[1];
    match words[1] {
        6 => {}
        7 => w[5] = 1,
        3 | 5 | 10 => (w[5], w[7]) = (1, 1),
        _ => w[5] = 1,
    }
    let mut desc = srv_desc(&w);
    desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    // SAFETY: a slot of our heap; a null resource is allowed with a desc.
    unsafe {
        host.device
            .CreateShaderResourceView(None::<&ID3D12Resource>, Some(&desc), at)
    };
}

/// A view that reads zeros, for slots whose view the host cannot make.
/// shortcut: slots with no view at all (kind unknown) get this null 2D
/// view; null buffer views there did not stop the world's low-address
/// page faults (2026-11-24, run w10), so they are not the cause.
fn null_view(at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    let Some(host) = host() else { return };
    let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
        Format: DXGI_FORMAT_R8G8B8A8_UNORM,
        ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
        Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
            Texture2D: D3D12_TEX2D_SRV {
                MipLevels: 1,
                ..Default::default()
            },
        },
    };
    // SAFETY: a slot of our heap; a null resource is allowed with a desc.
    unsafe {
        host.device
            .CreateShaderResourceView(None::<&ID3D12Resource>, Some(&desc), at)
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A null texture UAV keeps the title's dimension (3D for the world's
    /// wind field), and is 2D when the title names none.
    #[test]
    fn null_uavs_keep_the_titles_dimension() {
        let d = null_uav_desc(Some(8));
        assert_eq!(d.ViewDimension, D3D12_UAV_DIMENSION_TEXTURE3D);
        // SAFETY: the member the dimension names.
        assert_eq!(unsafe { d.Anonymous.Texture3D.WSize }, u32::MAX);
        let d = null_uav_desc(Some(5));
        assert_eq!(d.ViewDimension, D3D12_UAV_DIMENSION_TEXTURE2DARRAY);
        // SAFETY: the member the dimension names.
        assert_eq!(unsafe { d.Anonymous.Texture2DArray.ArraySize }, 1);
        let dimension = |d| null_uav_desc(d).ViewDimension;
        assert_eq!(dimension(Some(2)), D3D12_UAV_DIMENSION_TEXTURE1D);
        assert_eq!(dimension(Some(4)), D3D12_UAV_DIMENSION_TEXTURE2D);
        assert_eq!(dimension(None), D3D12_UAV_DIMENSION_TEXTURE2D);
        // Multisampled UAVs need a cap; they stay 2D.
        assert_eq!(dimension(Some(6)), D3D12_UAV_DIMENSION_TEXTURE2D);
    }

    /// A null 3D UAV made on the real device leaves it alive.
    #[test]
    fn null_3d_uav_is_accepted_by_the_device() {
        let Some(host) = host() else {
            return; // no host GPU
        };
        let Some(heap) = HostHeap::new(false, 4) else {
            return;
        };
        let view = View::Unordered(0, 0, Some([28, 8, 0, 0, 0, 0, 0, 0, 0, 0]));
        null_for(&view, heap.cpu(0));
        // SAFETY: a plain device query.
        assert!(unsafe { host.device.GetDeviceRemovedReason() }.is_ok());
    }

    #[test]
    fn free_space_counts_separate_runs_at_the_heap_ends() {
        assert_eq!(free_space(&[true; 8]), (0, 0));
        assert_eq!(free_space(&[false; 8]), (8, 8));
        let fragmented = [false, false, true, false, true, false, false, false];
        assert_eq!(free_space(&fragmented), (6, 3));
    }

    #[test]
    fn free_table_chooses_the_first_contiguous_fit_in_cursor_order() {
        for mask in 0u16..256 {
            let taken: [bool; 8] = std::array::from_fn(|slot| mask & (1 << slot) != 0);
            for cursor in 0..taken.len() {
                for n in 1..=taken.len() {
                    let expected = (cursor..taken.len()).chain(0..cursor).find(|&start| {
                        taken
                            .get(start..start + n)
                            .is_some_and(|table| table.iter().all(|&held| !held))
                    });
                    assert_eq!(
                        free_table(&taken, cursor, n),
                        expected,
                        "{mask:#x} {cursor} {n}"
                    );
                }
            }
        }
    }

    /// The CPU-side block used to initialize null samplers must fit the
    /// same portable sampler limit as the shader-visible heap.
    #[test]
    fn portable_sampler_null_fill_fits_the_host_limit() {
        if host().is_none() {
            return; // no host GPU
        }
        let heap = HostHeap::new(true, D3D12_MAX_SHADER_VISIBLE_SAMPLER_HEAP_SIZE)
            .expect("portable sampler heap must initialize");
        assert!(
            heap.fill_null(
                D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER,
                D3D12_MAX_SHADER_VISIBLE_SAMPLER_HEAP_SIZE,
            )
            .is_ok()
        );
    }

    /// Dynamic slots stay taken until given back, and runs do not wrap.
    #[test]
    fn dynamic_slots_are_held_until_released() {
        let Some(heap) = HostHeap::new(false, 4) else {
            return; // no host GPU
        };
        let a = heap.dynamic(DYNAMIC - 100).unwrap();
        let b = heap.dynamic(60).unwrap();
        assert_eq!((a, b), (4, 4 + DYNAMIC - 100));
        // 40 left at the end: 50 do not fit until something is given back.
        assert_eq!(heap.dynamic(50), None);
        heap.release(a, 50);
        assert_eq!(heap.dynamic(50), Some(4));
        assert_eq!(heap.dynamic(40), Some(4 + DYNAMIC - 40));
    }

    #[test]
    fn a_free_table_can_straddle_the_search_cursor() {
        let Some(heap) = HostHeap::new(false, 4) else {
            return;
        };
        let left = heap.dynamic(4).unwrap();
        let right = heap.dynamic(5).unwrap();
        let tail = heap.dynamic(DYNAMIC - 9).unwrap();

        heap.release(left, 4);
        assert_eq!(heap.dynamic(4), Some(left)); // cursor now separates the two tables
        heap.release(right, 5);
        heap.release(left, 4);
        assert_eq!(heap.dynamic(9), Some(left));
        assert_eq!(heap.dynamic(1), None, "the tail is still held");

        heap.release(tail, DYNAMIC - 9);
        assert_eq!(heap.dynamic(DYNAMIC - 9), Some(tail));
    }

    /// A view heap the host caps (the title asks for 1,310,720 views,
    /// NVIDIA takes 1,000,000) keeps all [`DYNAMIC`] dynamic slots: the
    /// title's slots give way, not the tables draws need made afresh.
    #[test]
    fn a_capped_view_heap_keeps_its_dynamic_slots() {
        let Some(heap) = HostHeap::new(false, 1_310_720) else {
            return; // no host GPU
        };
        assert_eq!(heap.spare, DYNAMIC);
        // Capped, unless the host takes the whole heap. The guard slot is
        // the heap's last, after the dynamic ones.
        if heap.count() != 1_310_720 {
            assert_eq!(heap.count(), PORTABLE_VIEWS - DYNAMIC - 1);
            assert_eq!(heap.guard_slot(), PORTABLE_VIEWS - 1);
        }
        assert_eq!(heap.dynamic(DYNAMIC), Some(heap.count()));
        assert_eq!(heap.dynamic(1), None);
    }

    /// A view heap's guard slot follows its dynamic slots and is never
    /// handed out; a sampler heap's is its last slot, dynamic ones included.
    #[test]
    fn the_guard_slot_is_past_every_dynamic_slot() {
        let Some(views) = HostHeap::new(false, 4) else {
            return; // no host GPU
        };
        assert_eq!(views.guard_slot(), 4 + DYNAMIC);
        let first = views.dynamic(DYNAMIC).unwrap();
        assert_eq!(first + DYNAMIC, views.guard_slot());
        assert_eq!(views.dynamic(1), None);

        let small = HostHeap::new(true, 16).unwrap();
        assert_eq!(small.guard_slot(), 15);
        let capped = HostHeap::new(true, 4 * SAMPLERS).unwrap();
        assert_eq!(capped.guard_slot(), SAMPLERS - 1);
    }

    /// A sampler heap past the host's limit is capped, not lost, and keeps
    /// dynamic slots for the tables reaching past what it holds.
    #[test]
    fn large_sampler_heap_is_capped() {
        let Some(heap) = HostHeap::new(true, 4 * SAMPLERS) else {
            return; // no host GPU
        };
        let count = SAMPLERS - SAMPLER_DYNAMIC;
        assert_eq!(heap.count(), count);
        assert_eq!(heap.dynamic(SAMPLER_DYNAMIC + 1), None);
        let a = heap.dynamic(SAMPLER_DYNAMIC).unwrap();
        assert_eq!(a, count);
        assert_eq!(heap.dynamic(1), None);
        heap.release(a, 8);
        assert_eq!(heap.dynamic(8), Some(count));
    }

    /// A sampler heap within the limit has a slot per title slot and no
    /// dynamic ones.
    #[test]
    fn small_sampler_heap_has_no_dynamic_slots() {
        let Some(heap) = HostHeap::new(true, 16) else {
            return; // no host GPU
        };
        assert_eq!(heap.count(), 16);
        assert_eq!(heap.dynamic(1), None);
    }

    /// A structured view whose start in its host copy is not on an element
    /// becomes the raw view of the same bytes, where those start 16-byte
    /// aligned.
    #[test]
    fn structured_views_off_an_element_are_raw_views_of_the_same_bytes() {
        // (offset in the copy, first element, elements, stride)
        for (offset, first, count, stride) in [
            (16, 0, 4, 32),
            (48, 3, 7, 32),
            (4096 + 16, 1, 1, 48),
            (32, 0, 9, 12),
        ] {
            let (dword, dwords) = raw_window(offset, first, count, stride).unwrap();
            assert_eq!(dword * 4, offset + first * u64::from(stride));
            assert_eq!(u64::from(dwords) * 4, u64::from(count) * u64::from(stride));
        }
        // Not 16-byte aligned, strides raw views cannot cover, too long.
        assert_eq!(raw_window(8, 0, 4, 32), None);
        assert_eq!(raw_window(16, 1, 4, 12), None);
        assert_eq!(raw_window(16, 0, 4, 6), None);
        assert_eq!(raw_window(16, 0, 4, 0), None);
        assert_eq!(raw_window(16, 0, u32::MAX, 32), None);
    }

    /// A structured view (32-byte elements) 16 bytes into its host copy,
    /// read and another written by a shader: it reads the title's bytes from
    /// there, and what it writes lands at its own bytes of title memory.
    #[test]
    fn structured_views_off_an_element_read_and_write_their_bytes() {
        use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
        use windows::Win32::Graphics::Direct3D::ID3DBlob;
        use windows::core::s;
        let Some(host) = host() else {
            return; // no host GPU
        };
        const SOURCE: &str = r#"
#define RS "DescriptorTable(SRV(t0), UAV(u0, numDescriptors = 2))"
ByteAddressBuffer view : register(t0);
RWByteAddressBuffer window : register(u0);
RWByteAddressBuffer seen : register(u1);
[RootSignature(RS)]
[numthreads(32, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    seen.Store(id.x * 4, view.Load(id.x * 4));
    window.Store(id.x * 4, 0xc0de0000u | id.x);
}
"#;
        let mut code: Option<ID3DBlob> = None;
        // SAFETY: the source and names live through the call; out pointer ours.
        unsafe {
            D3DCompile(
                SOURCE.as_ptr().cast(),
                SOURCE.len(),
                s!("views"),
                None,
                None,
                s!("main"),
                s!("cs_5_1"),
                0,
                0,
                &mut code,
                None,
            )
        }
        .unwrap();
        let code = code.unwrap();
        let device = &host.device;
        // SAFETY: the blob's own bytes, with the root signature inside;
        // plain object creation.
        let (root, pso, views, step) = unsafe {
            let bytes = std::slice::from_raw_parts(
                code.GetBufferPointer().cast::<u8>(),
                code.GetBufferSize(),
            );
            let root: ID3D12RootSignature = device.CreateRootSignature(0, bytes).unwrap();
            let pso: ID3D12PipelineState = device
                .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                    pRootSignature: std::mem::transmute_copy(&root),
                    CS: D3D12_SHADER_BYTECODE {
                        pShaderBytecode: bytes.as_ptr().cast(),
                        BytecodeLength: bytes.len(),
                    },
                    ..Default::default()
                })
                .unwrap();
            let views: ID3D12DescriptorHeap = device
                .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                    NumDescriptors: 3,
                    Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                    NodeMask: 0,
                })
                .unwrap();
            let step =
                device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
            (root, pso, views, step as usize)
        };
        // SAFETY: a live heap.
        let first = unsafe { views.GetCPUDescriptorHandleForHeapStart() }.ptr;
        let at = |i: usize| D3D12_CPU_DESCRIPTOR_HANDLE {
            ptr: first + i * step,
        };

        // A title buffer of 1 KiB, numbered dwords; four 32-byte elements
        // from byte 16 are read, four from byte 272 written.
        let before: Vec<u32> = (0..256).map(|i| 0x7171_0000 | i).collect();
        let mut title = before.clone();
        let start = title.as_mut_ptr() as u64;
        crate::shadow::register(start as usize, start, 1024);
        let shadow = crate::shadow::find(start, 1024).unwrap();
        let structured = D3D12_BUFFER_SRV {
            NumElements: 4,
            StructureByteStride: 32,
            ..Default::default()
        };
        let srv = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: DXGI_FORMAT_UNKNOWN,
            ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 { Buffer: structured },
        };
        let uav = |buffer| D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: DXGI_FORMAT_UNKNOWN,
            ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
            Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 { Buffer: buffer },
        };
        let written = D3D12_BUFFER_UAV {
            NumElements: 4,
            StructureByteStride: 32,
            ..Default::default()
        };
        // Views the host refuses are left null, as `HostHeap::mirror` leaves them.
        let mut placed = [0; 14];
        placed[0] = 1; // a buffer
        if buffer_srv(start + 16, 128, srv, at(0)).is_none() {
            null_for(&View::Placed(false, placed, [0; 16]), at(0));
        }
        if buffer_uav(start + 272, 128, uav(written), at(1)).is_none() {
            null_for(&View::Placed(true, placed, [0; 16]), at(1));
        }
        // What the shader read, as a raw view of 32 dwords of our own.
        let (seen, out) = host.shared_buffer(128).unwrap();
        let mut raw = uav(D3D12_BUFFER_UAV {
            NumElements: 32,
            Flags: D3D12_BUFFER_UAV_FLAG_RAW,
            ..Default::default()
        });
        raw.Format = DXGI_FORMAT_R32_TYPELESS;
        // SAFETY: a live buffer and a slot of our heap.
        unsafe {
            device.CreateUnorderedAccessView(&seen, None::<&ID3D12Resource>, Some(&raw), at(2))
        };

        let list = host.new_list().unwrap();
        shadow.refresh(&list, start, 1024);
        shadow.gpu_writes(&list, start + 272, 128);
        // SAFETY: an open list; our heap, root signature and pipeline.
        unsafe {
            list.list.SetDescriptorHeaps(&[Some(views.clone())]);
            list.list.SetComputeRootSignature(&root);
            list.list.SetPipelineState(&pso);
            list.list
                .SetComputeRootDescriptorTable(0, views.GetGPUDescriptorHandleForHeapStart());
            list.list.Dispatch(1, 1, 1);
        }
        list.close();
        host.execute(&[&list]);
        crate::shadow::unregister(start as usize, start);

        // SAFETY: 32 dwords the GPU wrote, read after it finished.
        let seen = unsafe { std::slice::from_raw_parts(out.cast::<u32>(), 32) };
        assert_eq!(seen, &before[4..36], "what the read view gave");
        let marks: Vec<u32> = (0..32).map(|i| 0xc0de_0000 | i).collect();
        assert_eq!(&title[68..100], &marks[..], "what the written view left");
        assert_eq!(
            (&title[..68], &title[100..]),
            (&before[..68], &before[100..])
        );
    }

    /// Typeless SRV and UAV views on typeless resources are remapped to the first
    /// concrete format of their family so D3D12 can create the view without E_INVALIDARG.
    #[test]
    fn typeless_view_on_typeless_resource_remaps_to_concrete_format() {
        // R8G8B8A8_TYPELESS (27) with R8G8B8A8_TYPELESS view (27) -> R8G8B8A8_UNORM (28)
        let mut fmt = DXGI_FORMAT(27);
        assert!(adapt_view_format(27, Some(&mut fmt), false));
        assert_eq!(fmt.0, 28);

        // B8G8R8A8_TYPELESS (90) with B8G8R8A8_TYPELESS view (90) -> B8G8R8A8_UNORM (87)
        let mut fmt = DXGI_FORMAT(90);
        assert!(adapt_view_format(90, Some(&mut fmt), false));
        assert_eq!(fmt.0, 87);

        // R32_TYPELESS (39) with R32_TYPELESS view (39) -> R32_FLOAT (41)
        let mut fmt = DXGI_FORMAT(39);
        assert!(adapt_view_format(39, Some(&mut fmt), false));
        assert_eq!(fmt.0, 41);

        // UNKNOWN view (0) on typeless texture -> first concrete format
        let mut fmt = DXGI_FORMAT(0);
        assert!(adapt_view_format(27, Some(&mut fmt), false));
        assert_eq!(fmt.0, 28);

        // Typed view (29 = R8G8B8A8_UNORM_SRGB) on typeless texture (27) stays 29
        let mut fmt = DXGI_FORMAT(29);
        assert!(adapt_view_format(27, Some(&mut fmt), false));
        assert_eq!(fmt.0, 29);

        // Typeless view (27) on typed texture (28) adopts texture's format (28)
        let mut fmt = DXGI_FORMAT(27);
        assert!(adapt_view_format(28, Some(&mut fmt), false));
        assert_eq!(fmt.0, 28);

        // Typeless texture with None format is refused
        assert!(!adapt_view_format(27, None, false));

        // Incompatible family is refused
        let mut fmt = DXGI_FORMAT(10);
        assert!(!adapt_view_format(27, Some(&mut fmt), false));

        // UAV with sRGB (29) converts to linear (28)
        let mut fmt = DXGI_FORMAT(29);
        assert!(adapt_view_format(27, Some(&mut fmt), true));
        assert_eq!(fmt.0, 28);
    }
}

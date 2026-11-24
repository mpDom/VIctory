//! Built-in host D3D12 spatial upscaler for `amd_fidelityfx_loader_xbox.dll`
//! and `amd_fidelityfx_upscaler_xbox.dll`.
//!
//! On Series S (and Series X when FidelityFX is active), the title renders its
//! 3D scene at a lower internal resolution (e.g. `1280x720`) and calls
//! `ffxDispatch` to upscale to the display resolution (`1920x1080`). Rather
//! than running the console's Xbox FSR compute shaders (which depend on
//! console-specific shader features and may output black), this module answers
//! the five `ffx*` exports directly and records a lightweight host D3D12
//! bilinear compute upscale from `resources[0]` (color) to `resources[6]`
//! (output).
//!
//! Enabled by default; set `VICTORY_FSR_SPATIAL=0` to disable and let the
//! title load its own Xbox FidelityFX DLLs.
#![allow(non_snake_case)]
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::ffi::c_void;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU32, Ordering};

use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
use windows::Win32::Graphics::Direct3D::ID3DBlob;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::core::s;

use crate::object::This;

type ReturnCode = u32;
const OK: ReturnCode = 0;
const ERROR: ReturnCode = 1;
const ERROR_PARAMETER: ReturnCode = 6;

/// `ffxApiHeader`: a description's type and the next in its chain.
#[repr(C)]
pub struct Header {
    kind: u64,
    next: *mut Header,
}

const CREATE_UPSCALE: u64 = 0x0001_0000;
const DISPATCH_UPSCALE: u64 = 0x0001_0001;
const QUERY_GET_UPSCALE_RATIO: u64 = 0x0001_0002;
const QUERY_GET_RENDER_RESOLUTION: u64 = 0x0001_0003;
const QUERY_JITTER_PHASE_COUNT: u64 = 0x0001_0004;
const QUERY_JITTER_OFFSET: u64 = 0x0001_0005;
const QUERY_GET_VERSIONS: u64 = 4;
const QUERY_GET_PROVIDER_VERSION: u64 = 6;

const HOST_UNORDERED_ACCESS: u32 = 0x8;
const HOST_NON_PIXEL_SHADER_RESOURCE: u32 = 0x40;

const OUTPUT: usize = 6;
const RING_SLOTS: u32 = 64;

/// `ffxCreateContextDescUpscale`.
#[repr(C)]
#[derive(Clone, Copy)]
struct CreateUpscale {
    kind: u64,
    next: *mut Header,
    flags: u32,
    max_render: [u32; 2],
    max_upscale: [u32; 2],
    message: *const c_void,
}
const _: () = assert!(size_of::<CreateUpscale>() == 48);

/// `FfxApiResource`: the resource, its description, its state.
#[repr(C)]
#[derive(Clone, Copy)]
struct Resource {
    resource: *mut c_void,
    description: [u32; 8],
    state: u32,
}
const _: () = assert!(size_of::<Resource>() == 48);

/// `ffxDispatchDescUpscale`.
#[repr(C)]
#[derive(Clone, Copy)]
struct DispatchUpscale {
    kind: u64,
    next: *mut Header,
    command_list: *mut c_void,
    /// Color, depth, motion vectors, exposure, reactive, transparency and
    /// composition, output.
    resources: [Resource; 7],
    jitter: [f32; 2],
    motion_vector_scale: [f32; 2],
    render_size: [u32; 2],
    upscale_size: [u32; 2],
    sharpening: bool,
    sharpness: f32,
    frame_time_delta: f32,
    pre_exposure: f32,
    reset: bool,
    camera: [f32; 4],
    flags: u32,
}
const _: () = assert!(size_of::<DispatchUpscale>() == 432);

#[repr(C)]
struct GetVersions {
    kind: u64,
    next: *mut Header,
    create_kind: u64,
    device: *mut c_void,
    count: *mut u64,
    ids: *mut u64,
    names: *mut *const i8,
}

#[repr(C)]
struct ProviderVersion {
    kind: u64,
    next: *mut Header,
    id: u64,
    name: *const i8,
}

#[repr(C)]
struct UpscaleRatio {
    kind: u64,
    next: *mut Header,
    quality_mode: u32,
    ratio: *mut f32,
}

#[repr(C)]
struct RenderResolution {
    kind: u64,
    next: *mut Header,
    display_width: u32,
    display_height: u32,
    quality_mode: u32,
    render_width: *mut u32,
    render_height: *mut u32,
}

#[repr(C)]
struct JitterPhaseCount {
    kind: u64,
    next: *mut Header,
    render_width: u32,
    display_width: u32,
    out: *mut i32,
}

#[repr(C)]
struct JitterOffset {
    kind: u64,
    next: *mut Header,
    index: i32,
    phase_count: i32,
    out_x: *mut f32,
    out_y: *mut f32,
}

/// Whether the built-in spatial upscaler hook is active (default on; set
/// `VICTORY_FSR_SPATIAL=0` to disable).
pub fn enabled() -> bool {
    std::env::var("VICTORY_FSR_SPATIAL")
        .map(|v| v != "0")
        .unwrap_or(true)
}

fn quality_ratio(mode: u32) -> f32 {
    match mode {
        0 => 1.0, // NativeAA
        1 => 1.5, // Quality (1920x1080 -> 1280x720)
        2 => 1.7, // Balanced
        3 => 2.0, // Performance
        4 => 3.0, // Ultra Performance
        _ => 1.5,
    }
}

/// Maps typeless DXGI formats to a concrete view format for SRV/UAV creation.
fn concrete_view_format(fmt: DXGI_FORMAT) -> DXGI_FORMAT {
    match fmt {
        DXGI_FORMAT_R16G16B16A16_TYPELESS => DXGI_FORMAT_R16G16B16A16_FLOAT,
        DXGI_FORMAT_R10G10B10A2_TYPELESS => DXGI_FORMAT_R10G10B10A2_UNORM,
        DXGI_FORMAT_R8G8B8A8_TYPELESS => DXGI_FORMAT_R8G8B8A8_UNORM,
        DXGI_FORMAT_B8G8R8A8_TYPELESS => DXGI_FORMAT_B8G8R8A8_UNORM,
        DXGI_FORMAT_R32G32B32A32_TYPELESS => DXGI_FORMAT_R32G32B32A32_FLOAT,
        other => other,
    }
}

struct SpatialPipeline {
    root: ID3D12RootSignature,
    pso: ID3D12PipelineState,
    heap: ID3D12DescriptorHeap,
    stride: u32,
    next_slot: AtomicU32,
}

impl SpatialPipeline {
    fn new(device: &ID3D12Device) -> Option<Self> {
        const SOURCE: &str = r#"
            #define RS "RootConstants(num32BitConstants=4, b0), \
                        DescriptorTable(SRV(t0), UAV(u0)), \
                        StaticSampler(s0, filter=FILTER_MIN_MAG_MIP_LINEAR, \
                            addressU=TEXTURE_ADDRESS_CLAMP, addressV=TEXTURE_ADDRESS_CLAMP, \
                            addressW=TEXTURE_ADDRESS_CLAMP)"
            cbuffer Params : register(b0) {
                uint2 renderSize;
                uint2 upscaleSize;
            };
            Texture2D<float4> src : register(t0);
            RWTexture2D<float4> dst : register(u0);
            SamplerState linearClamp : register(s0);

            [RootSignature(RS)]
            [numthreads(8, 8, 1)]
            void main(uint3 id : SV_DispatchThreadID) {
                if (id.x >= upscaleSize.x || id.y >= upscaleSize.y) return;
                uint srcW, srcH;
                src.GetDimensions(srcW, srcH);
                float2 scale = float2(renderSize) / float2(max(srcW, 1u), max(srcH, 1u));
                float2 uv = ((float2(id.xy) + 0.5) / float2(upscaleSize)) * scale;
                dst[id.xy] = src.SampleLevel(linearClamp, uv, 0.0);
            }
        "#;
        let (mut code, mut errors): (Option<ID3DBlob>, Option<ID3DBlob>) = (None, None);
        // SAFETY: static source and entry/target strings; out pointers ours.
        let compiled = unsafe {
            D3DCompile(
                SOURCE.as_ptr().cast(),
                SOURCE.len(),
                s!("ffx_spatial"),
                None,
                None,
                s!("main"),
                s!("cs_5_0"),
                0,
                0,
                &mut code,
                Some(&mut errors),
            )
        };
        if let Err(e) = compiled {
            let msg = errors
                .as_ref()
                .map(|b| unsafe {
                    let slice = std::slice::from_raw_parts(
                        b.GetBufferPointer().cast::<u8>(),
                        b.GetBufferSize(),
                    );
                    String::from_utf8_lossy(slice).into_owned()
                })
                .unwrap_or_default();
            tracing::error!(error = %e, msg, "ffx: spatial compute shader failed to compile");
            return None;
        }
        let blob = code?;
        // SAFETY: valid compiled DXBC blob from D3DCompile.
        let bytes = unsafe {
            std::slice::from_raw_parts(blob.GetBufferPointer().cast::<u8>(), blob.GetBufferSize())
        };
        // SAFETY: valid device and embedded root signature in DXBC blob.
        let root: ID3D12RootSignature = unsafe { device.CreateRootSignature(0, bytes) }.ok()?;
        let pso_desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
            pRootSignature: std::mem::ManuallyDrop::new(Some(root.clone())),
            CS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: bytes.as_ptr().cast(),
                BytecodeLength: bytes.len(),
            },
            ..Default::default()
        };
        // SAFETY: valid root signature and compute shader bytecode.
        let pso: ID3D12PipelineState = unsafe { device.CreateComputePipelineState(&pso_desc) }.ok()?;
        let heap_desc = D3D12_DESCRIPTOR_HEAP_DESC {
            Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
            NumDescriptors: RING_SLOTS * 2,
            Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
            NodeMask: 0,
        };
        // SAFETY: valid descriptor heap description.
        let heap: ID3D12DescriptorHeap = unsafe { device.CreateDescriptorHeap(&heap_desc) }.ok()?;
        // SAFETY: valid device.
        let stride = unsafe {
            device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV)
        };
        Some(Self {
            root,
            pso,
            heap,
            stride,
            next_slot: AtomicU32::new(0),
        })
    }

    fn dispatch(
        &self,
        device: &ID3D12Device,
        list: &ID3D12GraphicsCommandList,
        color: &ID3D12Resource,
        output: &ID3D12Resource,
        render_size: [u32; 2],
        upscale_size: [u32; 2],
    ) {
        let pair = self.next_slot.fetch_add(1, Ordering::Relaxed) % RING_SLOTS;
        let base_idx = pair * 2;
        // SAFETY: valid heap and resources.
        unsafe {
            let cpu_start = self.heap.GetCPUDescriptorHandleForHeapStart();
            let gpu_start = self.heap.GetGPUDescriptorHandleForHeapStart();
            let srv_cpu = D3D12_CPU_DESCRIPTOR_HANDLE {
                ptr: cpu_start.ptr + (base_idx * self.stride) as usize,
            };
            let uav_cpu = D3D12_CPU_DESCRIPTOR_HANDLE {
                ptr: cpu_start.ptr + ((base_idx + 1) * self.stride) as usize,
            };
            let table_gpu = D3D12_GPU_DESCRIPTOR_HANDLE {
                ptr: gpu_start.ptr + u64::from(base_idx * self.stride),
            };

            let src_desc = color.GetDesc();
            let dst_desc = output.GetDesc();
            let srv_desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
                Format: concrete_view_format(src_desc.Format),
                ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
                Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                    Texture2D: D3D12_TEX2D_SRV {
                        MostDetailedMip: 0,
                        MipLevels: 1,
                        PlaneSlice: 0,
                        ResourceMinLODClamp: 0.0,
                    },
                },
            };
            let uav_desc = D3D12_UNORDERED_ACCESS_VIEW_DESC {
                Format: concrete_view_format(dst_desc.Format),
                ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2D,
                Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                    Texture2D: D3D12_TEX2D_UAV {
                        MipSlice: 0,
                        PlaneSlice: 0,
                    },
                },
            };
            device.CreateShaderResourceView(color, Some(&srv_desc), srv_cpu);
            device.CreateUnorderedAccessView(output, None, Some(&uav_desc), uav_cpu);

            list.SetDescriptorHeaps(&[Some(self.heap.clone())]);
            list.SetComputeRootSignature(&self.root);
            list.SetPipelineState(&self.pso);
            let params = [
                render_size[0],
                render_size[1],
                upscale_size[0],
                upscale_size[1],
            ];
            list.SetComputeRoot32BitConstants(0, 4, params.as_ptr().cast(), 0);
            list.SetComputeRootDescriptorTable(1, table_gpu);
            list.Dispatch(upscale_size[0].div_ceil(8), upscale_size[1].div_ceil(8), 1);
        }
    }
}

fn pipeline() -> Option<&'static SpatialPipeline> {
    static PIPE: OnceLock<Option<SpatialPipeline>> = OnceLock::new();
    PIPE.get_or_init(|| {
        let host = crate::host::host()?;
        SpatialPipeline::new(&host.device)
    })
    .as_ref()
}

const PROVIDER_ID: u64 = 0x0003_0001_0000;
const PROVIDER_NAME: &[u8] = b"VIctory FSR Spatial\0";

pub extern "C" fn ffxCreateContext(
    context: *mut *mut c_void,
    desc: *mut Header,
    _memory: *const c_void,
) -> ReturnCode {
    if context.is_null() || desc.is_null() {
        return ERROR_PARAMETER;
    }
    let mut upscale = None;
    let mut h = desc;
    while !h.is_null() {
        // SAFETY: title's header chain.
        let kind = unsafe { (*h).kind };
        if kind == CREATE_UPSCALE {
            // SAFETY: description of type CREATE_UPSCALE.
            upscale = Some(unsafe { *h.cast::<CreateUpscale>() });
        }
        // SAFETY: as above.
        h = unsafe { (*h).next };
    }
    if let Some(u) = upscale {
        tracing::info!(
            render = ?u.max_render,
            upscale = ?u.max_upscale,
            flags = u.flags,
            "ffx: created built-in spatial upscale context"
        );
    }
    // SAFETY: checked non-null.
    unsafe { *context = 1usize as *mut c_void };
    OK
}

pub extern "C" fn ffxDestroyContext(
    context: *mut *mut c_void,
    _memory: *const c_void,
) -> ReturnCode {
    if !context.is_null() {
        // SAFETY: checked non-null.
        unsafe { *context = std::ptr::null_mut() };
    }
    OK
}

pub extern "C" fn ffxConfigure(_context: *mut *mut c_void, _desc: *const Header) -> ReturnCode {
    OK
}

pub extern "C" fn ffxQuery(_context: *mut *mut c_void, desc: *mut Header) -> ReturnCode {
    if desc.is_null() {
        return ERROR_PARAMETER;
    }
    // SAFETY: title's header.
    let kind = unsafe { (*desc).kind };
    match kind {
        QUERY_GET_VERSIONS => {
            // SAFETY: description of type QUERY_GET_VERSIONS.
            let q = unsafe { &*desc.cast::<GetVersions>() };
            if !q.count.is_null() {
                // SAFETY: title's out pointer.
                unsafe { *q.count = 1 };
            }
            if !q.ids.is_null() {
                // SAFETY: title's out pointer.
                unsafe { *q.ids = PROVIDER_ID };
            }
            if !q.names.is_null() {
                // SAFETY: title's out pointer.
                unsafe { *q.names = PROVIDER_NAME.as_ptr().cast() };
            }
            OK
        }
        QUERY_GET_PROVIDER_VERSION => {
            // SAFETY: description of type QUERY_GET_PROVIDER_VERSION.
            let q = unsafe { &mut *desc.cast::<ProviderVersion>() };
            q.id = PROVIDER_ID;
            q.name = PROVIDER_NAME.as_ptr().cast();
            OK
        }
        QUERY_GET_UPSCALE_RATIO => {
            // SAFETY: description of type QUERY_GET_UPSCALE_RATIO.
            let q = unsafe { &*desc.cast::<UpscaleRatio>() };
            if !q.ratio.is_null() {
                // SAFETY: title's out pointer.
                unsafe { *q.ratio = quality_ratio(q.quality_mode) };
            }
            OK
        }
        QUERY_GET_RENDER_RESOLUTION => {
            // SAFETY: description of type QUERY_GET_RENDER_RESOLUTION.
            let q = unsafe { &*desc.cast::<RenderResolution>() };
            let r = quality_ratio(q.quality_mode);
            let rw = ((q.display_width as f32) / r).round().max(1.0) as u32;
            let rh = ((q.display_height as f32) / r).round().max(1.0) as u32;
            if !q.render_width.is_null() {
                // SAFETY: title's out pointer.
                unsafe { *q.render_width = rw };
            }
            if !q.render_height.is_null() {
                // SAFETY: title's out pointer.
                unsafe { *q.render_height = rh };
            }
            OK
        }
        QUERY_JITTER_PHASE_COUNT => {
            // SAFETY: description of type QUERY_JITTER_PHASE_COUNT.
            let q = unsafe { &*desc.cast::<JitterPhaseCount>() };
            if !q.out.is_null() {
                // SAFETY: title's out pointer.
                unsafe { *q.out = 1 };
            }
            OK
        }
        QUERY_JITTER_OFFSET => {
            // SAFETY: description of type QUERY_JITTER_OFFSET.
            let q = unsafe { &*desc.cast::<JitterOffset>() };
            if !q.out_x.is_null() {
                // SAFETY: title's out pointer.
                unsafe { *q.out_x = 0.0 };
            }
            if !q.out_y.is_null() {
                // SAFETY: title's out pointer.
                unsafe { *q.out_y = 0.0 };
            }
            OK
        }
        _ => OK,
    }
}

pub extern "C" fn ffxDispatch(context: *mut *mut c_void, desc: *const Header) -> ReturnCode {
    if context.is_null() || desc.is_null() {
        return ERROR_PARAMETER;
    }
    // SAFETY: title's header.
    let kind = unsafe { (*desc).kind };
    if kind != DISPATCH_UPSCALE {
        return OK;
    }
    // SAFETY: description of type DISPATCH_UPSCALE.
    let d = unsafe { *desc.cast::<DispatchUpscale>() };
    let title_list = d.command_list as This;
    let Some(list) = crate::list::host_list(title_list) else {
        return ERROR_PARAMETER;
    };
    let color_title = d.resources[0].resource as This;
    let output_title = d.resources[OUTPUT].resource as This;
    if color_title.is_null() || output_title.is_null() || color_title == output_title {
        return ERROR_PARAMETER;
    }
    let (Some(color), Some(output)) = (
        crate::resource::host_resource(color_title),
        crate::resource::host_resource(output_title),
    ) else {
        return ERROR_PARAMETER;
    };
    let (Some(host), Some(pipe)) = (crate::host::host(), pipeline()) else {
        return ERROR;
    };
    crate::list::ensure_state(list, color_title, HOST_NON_PIXEL_SHADER_RESOURCE);
    crate::list::ensure_state(list, output_title, HOST_UNORDERED_ACCESS);

    let render_size = if d.render_size[0] > 0 && d.render_size[1] > 0 {
        d.render_size
    } else {
        // SAFETY: valid D3D12 resource.
        let desc = unsafe { color.GetDesc() };
        [desc.Width as u32, desc.Height]
    };
    let upscale_size = if d.upscale_size[0] > 0 && d.upscale_size[1] > 0 {
        d.upscale_size
    } else {
        // SAFETY: valid D3D12 resource.
        let desc = unsafe { output.GetDesc() };
        [desc.Width as u32, desc.Height]
    };

    pipe.dispatch(&host.device, &list.list, color, output, render_size, upscale_size);
    crate::list::forget_host_bindings(title_list);

    static FIRST: std::sync::Once = std::sync::Once::new();
    FIRST.call_once(|| {
        // SAFETY: valid D3D12 resources.
        let (src_fmt, dst_fmt) = unsafe { (color.GetDesc().Format.0, output.GetDesc().Format.0) };
        tracing::info!(
            render = ?render_size,
            upscale = ?upscale_size,
            src_fmt,
            dst_fmt,
            "ffx: spatial upscale recorded on host GPU"
        );
    });
    OK
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quality_mode_returns_720p_for_1080p_output() {
        let mut rw = 0u32;
        let mut rh = 0u32;
        let mut q = RenderResolution {
            kind: QUERY_GET_RENDER_RESOLUTION,
            next: std::ptr::null_mut(),
            display_width: 1920,
            display_height: 1080,
            quality_mode: 1,
            render_width: &mut rw,
            render_height: &mut rh,
        };
        assert_eq!(
            ffxQuery(std::ptr::null_mut(), (&raw mut q).cast()),
            OK
        );
        assert_eq!((rw, rh), (1280, 720));
    }

    #[test]
    fn jitter_queries_return_single_unbiased_phase() {
        let mut phases = 0i32;
        let mut q1 = JitterPhaseCount {
            kind: QUERY_JITTER_PHASE_COUNT,
            next: std::ptr::null_mut(),
            render_width: 1280,
            display_width: 1920,
            out: &mut phases,
        };
        assert_eq!(ffxQuery(std::ptr::null_mut(), (&raw mut q1).cast()), OK);
        assert_eq!(phases, 1);

        let (mut jx, mut jy) = (1.0f32, 1.0f32);
        let mut q2 = JitterOffset {
            kind: QUERY_JITTER_OFFSET,
            next: std::ptr::null_mut(),
            index: 0,
            phase_count: 1,
            out_x: &mut jx,
            out_y: &mut jy,
        };
        assert_eq!(ffxQuery(std::ptr::null_mut(), (&raw mut q2).cast()), OK);
        assert_eq!((jx, jy), (0.0, 0.0));
    }
}

//! `amd_fidelityfx_loader_xbox.dll` and `amd_fidelityfx_upscaler_xbox.dll`:
//! answered by VIctory's built-in host D3D12 spatial upscaler (`victory_gpu::ffx`)
//! by default (disable with `VICTORY_FSR_SPATIAL=0`).

use crate::{HleExport, HleModule};

pub(crate) fn module() -> HleModule {
    use victory_gpu::ffx;
    let export = |name, address: usize| HleExport { name, address };
    HleModule {
        name: "ffx",
        dlls: &[
            "amd_fidelityfx_loader_xbox.dll",
            "amd_fidelityfx_upscaler_xbox.dll",
        ],
        exports: vec![
            export(
                "ffxCreateContext",
                ffx::ffxCreateContext as *const () as usize,
            ),
            export(
                "ffxDestroyContext",
                ffx::ffxDestroyContext as *const () as usize,
            ),
            export("ffxConfigure", ffx::ffxConfigure as *const () as usize),
            export("ffxQuery", ffx::ffxQuery as *const () as usize),
            export("ffxDispatch", ffx::ffxDispatch as *const () as usize),
        ],
    }
}

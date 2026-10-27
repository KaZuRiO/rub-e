//! Compiled shader modules.

use log::info;

use crate::device::Device;

/// A compiled shader module. One module can back several pipelines, and can
/// hold several entry points.
pub struct Shader(wgpu::ShaderModule);

impl Shader {
    /// Compiles WGSL source, typically from `include_str!`.
    ///
    /// A syntax or validation error in the source surfaces as a panic from the
    /// backend rather than an `Err`: shaders are compiled in with the binary,
    /// so a bad one is a bug, not a runtime condition.
    pub fn wgsl(device: &Device, label: &str, source: &str) -> Self {
        info!("compiling shader '{label}'");
        Self(
            device
                .raw()
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some(label),
                    source: wgpu::ShaderSource::Wgsl(source.into()),
                }),
        )
    }

    pub(crate) fn raw(&self) -> &wgpu::ShaderModule {
        &self.0
    }
}

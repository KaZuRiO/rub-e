//! Compiled shader modules.

use log::info;

use crate::device::Device;

/// A compiled shader module. One module can back several pipelines, and can
/// hold several entry points.
pub struct Shader(wgpu::ShaderModule);

impl Shader {
    /// Compiles WGSL source, typically from `include_str!`.
    ///
    /// # Panics
    ///
    /// On a syntax or validation error in the source, with a message naming the
    /// shader and the backend's diagnostic, rather than returning an `Err`:
    /// shaders are compiled in with the binary, so a bad one is a bug, not a
    /// runtime condition.
    pub fn wgsl(device: &Device, label: &str, source: &str) -> Self {
        info!("compiling shader '{label}'");
        Self(device.checked("shader", label, |raw| {
            raw.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(source.into()),
            })
        }))
    }

    pub(crate) fn raw(&self) -> &wgpu::ShaderModule {
        &self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::{device, panic_message};

    #[test]
    fn an_invalid_shader_panics_with_its_label() {
        let Some(device) = device() else { return };
        let msg = panic_message(|| {
            Shader::wgsl(&device, "broken", "fn main( {");
        });
        assert!(msg.starts_with("shader 'broken' is invalid"), "{msg}");
    }
}

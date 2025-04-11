use liquid_core::{Result, LiquidError};
use ndarray::{Array1, Array2};
use tracing::{debug, info, warn};

mod cuda;
mod wgpu;
mod error;

pub use error::GpuError;

/// Supported GPU backends
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GpuBackend {
    Cuda,
    Vulkan,
    Metal,
    WebGPU,
}

/// GPU device configuration
#[derive(Debug, Clone)]
pub struct GpuConfig {
    pub backend: GpuBackend,
    pub device_id: usize,
    pub memory_limit: usize,
    pub enable_tensor_cores: bool,
    pub enable_profiling: bool,
}

impl Default for GpuConfig {
    fn default() -> Self {
        Self {
            backend: GpuBackend::WebGPU,
            device_id: 0,
            memory_limit: 8 * 1024 * 1024 * 1024, // 8GB
            enable_tensor_cores: true,
            enable_profiling: false,
        }
    }
}

/// Trait for GPU-accelerated operations
pub trait GpuAccelerated {
    fn to_gpu(&self) -> Result<GpuTensor>;
    fn from_gpu(tensor: &GpuTensor) -> Result<Self> where Self: Sized;
}

/// GPU tensor wrapper
#[derive(Debug)]
pub struct GpuTensor {
    data: Box<dyn GpuBuffer>,
    shape: Vec<usize>,
    dtype: DataType,
}

/// Trait for GPU memory buffers
pub trait GpuBuffer: std::fmt::Debug {
    fn as_ptr(&self) -> *const u8;
    fn as_mut_ptr(&mut self) -> *mut u8;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Supported data types for GPU tensors
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DataType {
    Float32,
    Float64,
    Int32,
    Int64,
}

impl GpuAccelerated for Array1<f64> {
    fn to_gpu(&self) -> Result<GpuTensor> {
        // Implementation depends on selected backend
        #[cfg(feature = "cuda")]
        {
            cuda::array_to_gpu(self)
        }
        #[cfg(not(feature = "cuda"))]
        {
            wgpu::array_to_gpu(self)
        }
    }

    fn from_gpu(tensor: &GpuTensor) -> Result<Self> {
        // Implementation depends on selected backend
        #[cfg(feature = "cuda")]
        {
            cuda::array_from_gpu(tensor)
        }
        #[cfg(not(feature = "cuda"))]
        {
            wgpu::array_from_gpu(tensor)
        }
    }
}

impl GpuAccelerated for Array2<f64> {
    fn to_gpu(&self) -> Result<GpuTensor> {
        // Implementation depends on selected backend
        #[cfg(feature = "cuda")]
        {
            cuda::array_to_gpu(self)
        }
        #[cfg(not(feature = "cuda"))]
        {
            wgpu::array_to_gpu(self)
        }
    }

    fn from_gpu(tensor: &GpuTensor) -> Result<Self> {
        // Implementation depends on selected backend
        #[cfg(feature = "cuda")]
        {
            cuda::array_from_gpu(tensor)
        }
        #[cfg(not(feature = "cuda"))]
        {
            wgpu::array_from_gpu(tensor)
        }
    }
}

/// Initialize GPU device
pub fn init_gpu(config: &GpuConfig) -> Result<()> {
    info!("Initializing GPU with config: {:?}", config);
    
    match config.backend {
        GpuBackend::Cuda => {
            #[cfg(feature = "cuda")]
            {
                cuda::init_device(config.device_id, config.memory_limit)?;
                if config.enable_tensor_cores {
                    cuda::enable_tensor_cores()?;
                }
            }
            #[cfg(not(feature = "cuda"))]
            {
                return Err(GpuError::UnsupportedBackend("CUDA support not enabled".into()).into());
            }
        }
        GpuBackend::WebGPU => {
            wgpu::init_device(config)?;
        }
        _ => {
            warn!("Backend {:?} not yet implemented, falling back to WebGPU", config.backend);
            wgpu::init_device(config)?;
        }
    }
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_init() {
        let config = GpuConfig::default();
        init_gpu(&config).unwrap();
    }

    #[test]
    fn test_array_transfer() {
        let config = GpuConfig::default();
        init_gpu(&config).unwrap();

        let arr = Array1::linspace(0., 9., 10);
        let gpu_tensor = arr.to_gpu().unwrap();
        let arr2 = Array1::from_gpu(&gpu_tensor).unwrap();

        assert_eq!(arr, arr2);
    }
} 
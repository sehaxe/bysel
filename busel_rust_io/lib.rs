use pyo3::prelude::*;
use pyo3::types::PyModule;
use pyo3::types::PyModuleMethods;
use rayon::prelude::*;
use std::fs::OpenOptions;
use std::io::Write;
use std::fs::File;
use memmap2::Mmap;

#[pyclass]
struct ByteStreamer {
    mmap: Mmap,
    _file: File, // 🎯 КРИТИЧЕСКИЙ СИСТЕМНЫЙ ФИКС: Храним файл открытым, чтобы mmap оставался валидным на macOS!
    position: usize,
    chunk_size: usize,
}

#[pymethods]
impl ByteStreamer {
    #[new]
    fn new(file_path: String, chunk_size: usize, start_offset: usize) -> PyResult<Self> {
        let file = File::open(file_path)?;
        let mmap = unsafe { Mmap::map(&file)? };

        Ok(ByteStreamer {
            mmap,
            _file: file,
            position: start_offset,
            chunk_size,
        })
    }

    fn next_chunk(&mut self, _py: Python) -> Option<Vec<u8>> {
        if self.position >= self.mmap.len() {
            return None;
        }

        let start = self.position;
        let end = std::cmp::min(self.position + self.chunk_size, self.mmap.len());
        
        let mut chunk = self.mmap[start..end].to_vec();

        if chunk.len() < self.chunk_size {
            chunk.resize(self.chunk_size, 0u8);
        }

        self.position = end;
        Some(chunk)
    }

    fn get_position(&self) -> usize {
        self.position
    }

    fn get_file_size(&self) -> usize {
        self.mmap.len()
    }

    fn get_progress(&self) -> f64 {
        if self.mmap.len() == 0 {
            return 100.0;
        }
        (self.position as f64 / self.mmap.len() as f64) * 100.0
    }
}

// 🎯 СВЕРХБЫСТРЫЙ И КОМПАКТНЫЙ ТЕРНАРНЫЙ ИНФЕРЕНС НА RUST (ОБНОВЛЕН ДЛЯ PyO3 0.28)
#[pyfunction]
fn ternary_matmul_cpu(
    py: Python,
    x: Vec<f32>,       // Вектор активаций [cols]
    w: Vec<i8>,        // Плоская матрица весов [rows * cols]
    rows: usize,
    cols: usize,
) -> PyResult<Vec<f32>> {
    py.detach(|| {
        let mut y = vec![0.0; rows];
        
        // Параллельный расчет каждой строки матрицы весов на всех ядрах процессора
        y.par_iter_mut().enumerate().for_each(|(i, val)| {
            let offset = i * cols;
            let row_w = &w[offset..offset + cols];
            let mut sum = 0.0;
            
            for j in 0..cols {
                let weight = row_w[j];
                if weight == 1 {
                    sum += x[j];
                } else if weight == -1 {
                    sum -= x[j];
                }
            }
            *val = sum;
        });
        
        Ok(y)
    })
}

// 🎯 СВЕРХБЫСТРЫЙ И КОМПАКТНЫЙ РЕКОРДЕР ДАТАСЕТОВ НА RUST
#[pyfunction]
fn append_to_binary_file(filepath: String, data: Vec<u8>) -> PyResult<()> {
    let mut file = OpenOptions::new()
        .create(true)
        .write(true)
        .append(true)
        .open(filepath)?;
    file.write_all(&data)?;
    Ok(())
}

#[pyfunction]
fn init_thread_pool(num_threads: usize) -> PyResult<()> {
    if num_threads > 0 {
        let _ = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .thread_name(|i| format!("busel-io-{}", i))
            .build_global();
    }
    Ok(())
}

#[pyfunction]
fn get_cpu_count() -> usize {
    std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(1)
}

#[pymodule]
fn busel(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<ByteStreamer>()?;
    m.add_function(wrap_pyfunction!(init_thread_pool, m)?)?;
    m.add_function(wrap_pyfunction!(get_cpu_count, m)?)?;
    m.add_function(wrap_pyfunction!(ternary_matmul_cpu, m)?)?;
    m.add_function(wrap_pyfunction!(append_to_binary_file, m)?)?;
    Ok(())
}
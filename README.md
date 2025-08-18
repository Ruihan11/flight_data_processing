# Flight Data Processing Benchmark

This project benchmarks **CPU serial processing** vs **CPU parallel processing** for extracting features from large-scale aircraft trajectory data.


## Impact
- **Efficiency**: Faster feature generation from raw flight data (x2.33) 
- **Business value**: Shorter cycle for model training and analysis  
- **Cost benefit**: Efficient CPU-only solution, suitable where GPUs are limited  

---

## Usage
Run CPU serial mode:
```bash
bash run_benchmark.sh --help

options:
  --dataset NAME        
  --sample-size SIZE    
  --min-rows ROWS       
  --mode MODE           
  -h, --help       

MODE:
  cpu           
  cpu_parallel  
  benchmark     

sample:
  ./run_benchmark.sh --mode cpu
  ./run_benchmark.sh --mode cpu_parallel
  ./run_benchmark.sh --mode benchmark
```

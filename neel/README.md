## What did I learn?
* Python decorators:
  * Example : @parser.wrap()
  * Can be used to wrap classes and functions to add certain functionality to them
  * @decorator on def func() is equivalent to func = decorator(func)
* @dataclass is a convenient decorator for classes which store data
* ** Separator to separate key value pairs of a python dictionary
* Use time.perf_counter() for measuring durations of loops
* Processors are convenient utils used to map teleoperator <> robot <> policy
  * Apply a pipeline to raw data 
* @safe_stop_image_writer decorator to handler clean up if a function throws
* Rerun visualisation to visualise camera frames and motor positions
* Dataset is created locally and then pushed to hugging face with HF python API
* Mixed precision training using torch.autocast() because 16 bit floats operations can be faster on GPUs, most float operations in pytorch are 32 bit
* GradScaler helps to ensure we dont have the vanishing gradient problem
* Torch optimizations for training
   * torch.backends.cudnn.benchmark : Benchmark different convolutional operations on your hardware and then pick the best one
   * torch.backends.cuda.matmul.allow_tf32 : tf32 math is faster on some new nvidia GPUs
* Created a torch data loader from lerobot dataset for training

import torch
import torch.autograd.profiler

def main():
  # Create a tensor
  x = torch.randn(100, 100)

  # Define a function
  def f(x):
    return torch.nn.functional.relu(x)

  # Profile the function
  with torch.autograd.profiler.profile() as prof:
    y = f(x)

  # Save the profiling results
  prof.export_chrome_trace("trace.json")
  

if __name__ == "__main__":
  main()


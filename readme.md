# FPGA CNN Accelerator Architecture

This project implements a small Convolutional Neural Network for image classification using Verilog on an FPGA.

The main goal is to build the CNN inference pipeline directly in hardware and understand how convolution, pooling, activation, fully connected layers, memory, and control logic are mapped onto FPGA resources.

The architecture is intentionally kept small so that every part of the design can be understood, simulated, and verified independently.

---

## CNN Architecture

The network uses the following structure:

```text
Input Image
28 × 28 × 1
      │
      ▼
Conv1
3 × 3 kernel
8 filters
Stride = 1
Padding = 1
      │
      ▼
28 × 28 × 8
      │
      ▼
ReLU
      │
      ▼
Max Pool
2 × 2
Stride = 2
      │
      ▼
14 × 14 × 8
      │
      ▼
Conv2
3 × 3 kernel
8 input channels
16 filters
Stride = 1
Padding = 1
      │
      ▼
14 × 14 × 16
      │
      ▼
ReLU
      │
      ▼
Max Pool
2 × 2
Stride = 2
      │
      ▼
7 × 7 × 16
      │
      ▼
Flatten
      │
      ▼
784 values
      │
      ▼
Fully Connected
784 → 10
      │
      ▼
10 Class Scores
      │
      ▼
Argmax
      │
      ▼
Predicted Class
0–9
```

---

## Layer Summary

| Layer           | Input        | Operation                   | Output          |
| --------------- | ------------ | --------------------------- | --------------- |
| Input           | —            | Grayscale image             | 28 × 28 × 1     |
| Conv1           | 28 × 28 × 1  | 3 × 3, 8 filters, S=1, P=1  | 28 × 28 × 8     |
| ReLU1           | 28 × 28 × 8  | `max(0, x)`                 | 28 × 28 × 8     |
| Pool1           | 28 × 28 × 8  | 2 × 2 max pool, S=2         | 14 × 14 × 8     |
| Conv2           | 14 × 14 × 8  | 3 × 3, 16 filters, S=1, P=1 | 14 × 14 × 16    |
| ReLU2           | 14 × 14 × 16 | `max(0, x)`                 | 14 × 14 × 16    |
| Pool2           | 14 × 14 × 16 | 2 × 2 max pool, S=2         | 7 × 7 × 16      |
| Flatten         | 7 × 7 × 16   | Reshape                     | 784             |
| Fully Connected | 784          | 784 × 10                    | 10 scores       |
| Argmax          | 10 scores    | Find largest score          | Predicted class |

---

## Hardware Architecture

The FPGA implementation separates the CNN into a datapath and a control path.

```text
                         cnn_top
┌───────────────────────────────────────────────────┐
│                                                   │
│                 ┌──────────────┐                  │
│                 │ Controller   │                  │
│                 │ FSM          │                  │
│                 └──────┬───────┘                  │
│                        │                          │
│                        ▼                          │
│                 Address / Control                 │
│                        │                          │
│                        ▼                          │
│              ┌──────────────────┐                 │
│              │ Feature Memory   │                 │
│              │ BRAM / RAM       │                 │
│              └────────┬─────────┘                 │
│                       │                           │
│                       ▼                           │
│              ┌──────────────────┐                 │
│              │ Window Generator │                 │
│              │ + Line Buffers   │                 │
│              └────────┬─────────┘                 │
│                       │                           │
│                  3 × 3 Window                     │
│                       │                           │
│                       ▼                           │
│              ┌──────────────────┐                 │
│              │ MAC Engine       │◄──── Weights   │
│              │ Multiply + Add   │                 │
│              └────────┬─────────┘                 │
│                       │                           │
│                       ▼                           │
│              ┌──────────────────┐                 │
│              │ Accumulator      │                 │
│              └────────┬─────────┘                 │
│                       │                           │
│                    + Bias                         │
│                       │                           │
│                       ▼                           │
│                     ReLU                          │
│                       │                           │
│                       ▼                           │
│                    Max Pool                       │
│                       │                           │
│                       ▼                           │
│               Feature Memory                     │
│                       │                           │
│                       ▼                           │
│                  FC Engine                        │
│                       │                           │
│                       ▼                           │
│                   10 Scores                       │
│                       │                           │
│                       ▼                           │
│                    Argmax                         │
│                       │                           │
└───────────────────────┼───────────────────────────┘
                        │
                        ▼
                 Predicted Class
```

---

## Convolution Engine

The convolution engine performs the main computation in the CNN.

For each output value:

```text
output =
bias
+
Σ(input_pixel × weight)
```

For a 3 × 3 convolution with one input channel:

```text
p0 × w0
p1 × w1
p2 × w2
p3 × w3
p4 × w4
p5 × w5
p6 × w6
p7 × w7
p8 × w8
```

The products are summed using an adder tree.

```text
p0*w0 ─┐
p1*w1 ─┤
p2*w2 ─┤
p3*w3 ─┤
p4*w4 ─┼──► Adder Tree ─► Partial Sum
p5*w5 ─┤
p6*w6 ─┤
p7*w7 ─┤
p8*w8 ─┘
```

The initial implementation uses a 9-way MAC architecture so that one complete 3 × 3 channel operation can be computed in parallel.

---

## Multi-Channel Convolution

Conv2 receives an input tensor of:

```text
14 × 14 × 8
```

Each Conv2 filter therefore has dimensions:

```text
3 × 3 × 8
```

One output pixel requires:

```text
3 × 3 × 8 = 72
```

multiply-accumulate operations.

The architecture processes one channel at a time using the 9-way MAC engine.

```text
Channel 0
3 × 3
   │
   ▼
9 MACs
   │
   ▼
Partial Sum
   │
   ▼
Accumulator
   ▲
   │
Channel 1
3 × 3
   │
   ▼
9 MACs
   │
   ▼
Partial Sum

...

Channel 7
   │
   ▼
Partial Sum
   │
   ▼
Final Accumulation
   │
   ▼
+ Bias
   │
   ▼
ReLU
```

---

## Window Generator

The convolution engine requires a 3 × 3 group of neighboring pixels.

Reading all nine pixels from memory for every convolution would be inefficient.

The design therefore uses line buffers and shift registers.

```text
Previous Row 1  ───────────────
Previous Row 2  ───────────────
Current Row     ───────────────
```

These buffers continuously maintain the pixel data required to create a moving 3 × 3 window.

```text
x0 x1 x2
x3 x4 x5
x6 x7 x8
```

As new pixels arrive, the window shifts across the image.

This allows pixel data to be reused instead of repeatedly reading the same values from memory.

---

## ReLU

The ReLU activation is:

```text
ReLU(x) = max(0, x)
```

In hardware:

```text
if x < 0
    output = 0
else
    output = x
```

For signed two's-complement values, the sign bit can be checked directly.

ReLU does not change the dimensions of the feature map.

---

## Max Pooling

The design uses:

```text
2 × 2 Max Pool
Stride = 2
```

Example:

```text
1 7
3 4
```

becomes:

```text
7
```

A 2 × 2 pool can be implemented using three comparators.

```text
a ─┐
   ├─ max ─┐
b ─┘       │
           ├─ max ─► output
c ─┐       │
   ├─ max ─┘
d ─┘
```

Pooling reduces the spatial dimensions:

```text
28 × 28 × 8
      ↓
14 × 14 × 8
```

and later:

```text
14 × 14 × 16
      ↓
7 × 7 × 16
```

---

## Flatten Layer

The final pooled feature map has dimensions:

```text
7 × 7 × 16
```

which contains:

```text
7 × 7 × 16 = 784
```

values.

Flattening does not require mathematical computation.

The values are simply treated as a one-dimensional vector:

```text
x[0]
x[1]
x[2]
...
x[783]
```

---

## Fully Connected Layer

The fully connected layer converts the 784 feature values into 10 class scores.

```text
784 inputs
    │
    ▼
10 output neurons
```

Each output neuron performs:

```text
score[j] =
bias[j]
+
Σ(input[i] × weight[j][i])
```

The layer therefore contains:

```text
784 × 10 = 7,840
```

weights.

The same MAC hardware used for convolution can potentially be reused for the fully connected layer.

---

## Classification

The final layer produces 10 scores:

```text
score_0
score_1
score_2
score_3
score_4
score_5
score_6
score_7
score_8
score_9
```

A hardware Softmax implementation is not required for basic classification.

Instead, an Argmax unit finds the index of the largest score.

```text
scores
  │
  ▼
Argmax
  │
  ▼
largest index
```

For example:

```text
[2, -1, 4, 7, 3, 1, 0, 12, 5, 6]
```

returns:

```text
7
```

so the predicted digit is `7`.

---

## Numerical Representation

The initial architecture uses integer quantization rather than floating-point arithmetic.

Recommended format:

```text
Weights      : signed INT8
Activations  : signed INT8
Products     : signed INT16
Accumulator  : signed INT32
```

Example:

```text
8-bit pixel
     ×
8-bit weight
     │
     ▼
16-bit product
     │
     ▼
32-bit accumulator
```

A wider accumulator is required because many multiplication results are summed together.

---

## Train in Python, Infer in Verilog

Training is performed outside the FPGA.

```text
Training Dataset
      │
      ▼
Python CNN
      │
      ▼
Trained Weights + Biases
      │
      ▼
Quantization
      │
      ▼
.mem Files
      │
      ▼
FPGA
      │
      ▼
Inference
```

The FPGA performs only the forward inference pass.

---

## Repository Structure

```text
cnn_fpga/
│
├── rtl/
│   ├── cnn_top.v
│   ├── controller.v
│   ├── conv_engine.v
│   ├── mac_unit.v
│   ├── mac_array.v
│   ├── accumulator.v
│   ├── line_buffer.v
│   ├── window_generator.v
│   ├── relu.v
│   ├── maxpool2x2.v
│   ├── fc_engine.v
│   ├── argmax.v
│   ├── feature_memory.v
│   └── weight_memory.v
│
├── tb/
│   ├── tb_mac_unit.v
│   ├── tb_conv_engine.v
│   ├── tb_relu.v
│   ├── tb_maxpool2x2.v
│   ├── tb_fc_engine.v
│   └── tb_cnn_top.v
│
├── python/
│   ├── train.py
│   ├── quantize.py
│   ├── export_weights.py
│   ├── golden_model.py
│   └── compare_results.py
│
├── mem/
│   ├── conv1_weights.mem
│   ├── conv1_bias.mem
│   ├── conv2_weights.mem
│   ├── conv2_bias.mem
│   ├── fc_weights.mem
│   └── fc_bias.mem
│
└── constraints/
    └── fpga.xdc
```

### `rtl/`

Contains the synthesizable Verilog hardware modules.

### `tb/`

Contains testbenches used to simulate and verify each hardware module.

### `python/`

Contains the software reference implementation, training code, quantization tools, and scripts for exporting trained parameters.

### `mem/`

Contains memory initialization files holding trained weights and biases.

---

## CNN Parameters

### Conv1

```text
Kernel Size      = 3 × 3
Input Channels   = 1
Output Filters   = 8
```

Weights:

```text
3 × 3 × 1 × 8 = 72
```

Biases:

```text
8
```

Total:

```text
80 parameters
```

### Conv2

```text
Kernel Size      = 3 × 3
Input Channels   = 8
Output Filters   = 16
```

Weights:

```text
3 × 3 × 8 × 16 = 1,152
```

Biases:

```text
16
```

Total:

```text
1,168 parameters
```

### Fully Connected

Weights:

```text
784 × 10 = 7,840
```

Biases:

```text
10
```

Total:

```text
7,850 parameters
```

### Total Network Parameters

```text
80 + 1,168 + 7,850
=
9,098 parameters
```

---

## Approximate Compute Cost

### Conv1

```text
28 × 28 × 8 × 3 × 3
=
56,448 MACs
```

### Conv2

```text
14 × 14 × 16 × 3 × 3 × 8
=
225,792 MACs
```

### Fully Connected

```text
784 × 10
=
7,840 MACs
```

### Total

```text
290,080 MACs per inference
```

Conv2 represents the majority of the computation and is therefore the main target for future parallelization.

---

## Verification Strategy

Every hardware module is tested independently before integrating the complete CNN.

```text
MAC Unit
   ↓
Convolution Engine
   ↓
ReLU
   ↓
Max Pool
   ↓
Conv1
   ↓
Conv2
   ↓
Fully Connected
   ↓
Argmax
   ↓
Complete CNN
```

A Python golden model generates the expected outputs.

The output of every RTL stage can then be compared with the Python result.

```text
Python Input
       │
       ├─────────────► Python Conv1
       │                     │
       │                     ▼
       │                 Expected
       │
       ▼
Verilog Conv1
       │
       ▼
Actual Result
       │
       ▼
Compare
```

Intermediate feature maps should be verified before testing the final prediction.

---

## Future Improvements

Once the baseline architecture is functional, future versions can explore:

* More parallel MAC units
* Multiple output filters processed simultaneously
* Better pipelining
* BRAM-based line buffers
* DSP48 utilization
* Weight reuse
* Double-buffered feature memories
* AXI4-Stream interfaces
* AXI DMA
* DDR memory support
* Higher clock frequency
* INT4 or binary quantization
* Convolution/FC hardware reuse
* Performance-per-watt analysis
* FPGA power side-channel analysis

---

## Project Goal

The purpose of this project is not simply to run a neural network on an FPGA.

The objective is to understand how a CNN algorithm becomes a hardware architecture:

```text
CNN Algorithm
      ↓
Loop Analysis
      ↓
Dataflow
      ↓
Memory Architecture
      ↓
MAC Parallelism
      ↓
Controller FSM
      ↓
RTL
      ↓
Simulation
      ↓
Synthesis
      ↓
FPGA Inference
```

The final system should provide a working image classifier while serving as a platform for studying FPGA accelerator design, hardware/software co-design, memory optimization, and parallel neural-network computation.

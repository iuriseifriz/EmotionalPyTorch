# EmotionalPyTorch
Simple AI model made with PyTorch to recognize emotions in text.

The program import a text dataset, tokenize it, creates the model, train it and run it in terminal.
The model is very limited due to the simple dataset I myself created to test it, but it can get much more smarter by the use of bigger dataset and adjusting the hyperparameters in more powerfull hardwares, thanks to the pytorch tensors.
Tests were made with HuggingFace free datasets, auto-augmentation and torchtext tokenizer, increasing not much the capacity of the final model by the expense of much more complexity. Due to that, the script was pruned and those parts were ripped off in the 03 version.

python 3.11.0
torch 2.0.1 
torchtext 0.15.2 (deprected)
numpy-1.26.4

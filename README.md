# LSTM

## The Challenge

Train an LSTM to solve the XOR problem: that is, given a sequence of bits, determine its parity. The LSTM should consume the sequence, one bit at a time, and then output the correct answer at the sequence’s end. Test the two approaches below:

Generate a dataset of random 100,000 binary strings of length 50. Train the LSTM; what performance do you get?
Generate a dataset of random 100,000 binary strings, where the length of each string is independently and randomly chosen between 1 and 50. Train the LSTM. Does it succeed? What explains the difference?

## How to run

Run main.py to train the LSTM on the parity prediction task.

## Example Output

```
RUNNING TRAINING...
EPOCH: 1
 -- For step: 0 accuracy is: 0.472
 -- For step: 200 accuracy is: 0.495
 -- For step: 400 accuracy is: 0.535
 -- For step: 600 accuracy is: 0.632
 -- For step: 800 accuracy is: 1.000
FULL ACCURACY REACHED: STOPPING
```

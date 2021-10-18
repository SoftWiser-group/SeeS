INPUT_DIM = 2
HID_DIM = 8
PF_DIM = 16
N_LAYERS = 2
N_HEADS = 2
SEQ_LEN = 5
WALK_LEN = 10
N_NODE = -1
BATCH_SIZE = 32
OUT_DIM = 1
LEARNING_RATE = 0.0001
MAX_EPOCHES = 300
SEED = 123
CLIP = 1
PATIENCE = 10



'''
Testing loss on SEQ 0: 7.8887062458917985
Testing loss on SEQ 1: 8.541566611261539
Testing loss on SEQ 2: 8.967023191526602
Testing loss on SEQ 3: 9.297423471881896
Testing loss on SEQ 4: 9.487563781763743
-------------------------------------------------
Data: traffic, Parameters: [2, 2, 8, 50]

Testing loss on SEQ 0: 6.614623345608583
Testing loss on SEQ 1: 7.569750809424621
Testing loss on SEQ 2: 8.239063879605435
Testing loss on SEQ 3: 8.731743647688242
Testing loss on SEQ 4: 9.056872495694432
-------------------------------------------------
Data: traffic, Parameters: [2, 2, 8, 100]

Testing loss: 0.13546666502952576
-------------------------------------------------
Data: finacial, Parameters: [2, 4, 8, 50]
Testing loss: 0.1344885677099228
-------------------------------------------------
Data: finacial, Parameters: [2, 8, 8, 50]
Terminating... No more improvement...
Testing...
Time: 0m 0s
Testing loss: 0.1362086981534958
-------------------------------------------------
Data: finacial, Parameters: [2, 2, 8, 50]

Testing loss: 0.13745072484016418
-------------------------------------------------
Data: finacial, Parameters: [3, 2, 8, 50]

Testing loss: 0.1370123028755188
-------------------------------------------------
Data: finacial, Parameters: [4, 2, 8, 50]

Testing loss: 0.13819919526576996
-------------------------------------------------
Data: finacial, Parameters: [5, 2, 8, 50]

8 0.13290542364120483
16 0.1345386654138565
32 0.13490475714206696
64 0.13341009616851807
128 0.13095353543758392
'''
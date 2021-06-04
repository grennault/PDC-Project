# ############################################################################
# main.py
# This is implementation of a Hadamard OR Random coding scheme 
# Example of running python script : 
# During oral presentation for COM 301 we will run default.
# python main.py                                                     (default is hadamard coding scheme with order 2**9 and using online server channel)
# python main.py --send_file msg-to-send.txt --receive_file msg-received.txt --server local  --msg_length 8 --block_length 370 --coding_scheme random
# python main.py --send_file msg-to-send.txt --receive_file msg-received.txt --server local  --msg_length 9 --block_length 512 --coding_scheme hadamard
# python main.py --send_file msg-to-send.txt --receive_file msg-received.txt --server online  --msg_length 8 --block_length 370 --coding_scheme random
# python main.py --send_file msg-to-send.txt --receive_file msg-received.txt --server online  --msg_length 9 --block_length 512 --coding_scheme hadamard
# Arguments :
# --send_file     : txt file with UTF-8 characters to transmit (max 80 bytes). Default is msg-to-send.txt.
# --receive_file  : txt file with UTF-8 characters received. Default is msg-received.txt.
# --server        : Channel server to use either 'local' or 'online' (ATTENTION : NEED TO BE CONNECTED TO EPFL VPN). Default is local.
# --msg_length    : Message lenght. Default is 9.
# --block_length  : Block lenght. If coding_scheme is hadamard must be equal to 2**msg_length. Default is 512.
# --coding_scheme : Coding scheme either 'random' or 'hadamard'. Default is hadamard.
# ================================================================
# Authors : Gaiëtan RENAULT - Sophia ARTIOLI - Alexandre SANTANGELO
# ############################################################################

import argparse, sys, numpy as np, pathlib, random, os, time

# -------------------- Helper functions --------------------
def parse_args():
    parser = argparse.ArgumentParser(description="COM-302 Project",
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     epilog="Transmit UTF-8 characters over noisy channel specified in assignement")
    parser.add_argument('--send_file', type=str, required=False, default='msg-to-send.txt',
                        help='.txt file containing 80 UTF-8 characters / byte to send')
    parser.add_argument('--receive_file', type=str, required=False, default='msg-received.txt',
                        help='.txt file where received characters will be stored')
    parser.add_argument('--server', type=str, required=False, default='local',
                        help='local or online')
    parser.add_argument('--msg_length', type=int, required=False, default='9',
                        help='Message length')
    parser.add_argument('--block_length', type=int, required=False, default='512',
                        help='Block length')
    parser.add_argument('--coding_scheme', type=str, required=False, default='hadamard',
                        help='hadamard or random')
    args = parser.parse_args()
    args.send_file = pathlib.Path(args.send_file).resolve(strict=True)
    if not (args.send_file.is_file() and
            (args.send_file.suffix == '.txt')):
        raise ValueError('Parameter[send_file] is not a .txt file.')
    args.receive_file = pathlib.Path(args.receive_file).resolve(strict=True)
    if not (args.receive_file.suffix == '.txt'):
        raise ValueError('Parameter[receive_file] is not a .txt file.')
    if args.msg_length > args.block_length or args.block_length <= 0 or args.msg_length <= 0 or \
            (args.coding_scheme == 'hadamard' and 2**args.msg_length != args.block_length):
        raise ValueError('Parameters[block_length, msg_length] are impossible')
    if args.server != 'local' and  args.server != 'online':
        raise ValueError('Parameter[server] must be local or online')
    if args.coding_scheme != 'hadamard' and  args.coding_scheme != 'random':
        raise ValueError('Parameter[coding-scheme] must be hadamard or random')
    return args
def arrayToString1D(array):
    return ''.join(map(str, array))
def arrayToStringAlongAxis(array, axis):
    return np.apply_along_axis(arrayToString1D, axis, array)
def convertElemArrayToBinary(array):
    return ((array.reshape(-1,1) & (2**np.arange(n))[::-1]) != 0).astype(int)
def blockPrint():
    sys.stdout = open(os.devnull, 'w')
def enablePrint():
    sys.stdout = sys.__stdout__
def zeroPaddingwithNull(binaryMsg): # To pad binaryMsg with 0x00
    x = 0
    while x < 80*8:                 # Compute x, lowest multiple of n greater than 640 bits
       x += n
    return binaryMsg.ljust(x, '0')
# -----------------------------------------------------------

# -------------------- Transmitter apply encoding scheme on msgToSend (txt file) and save result in input.txt --------------------
def transmitter(msgToSend, coding_scheme):
    with open(msgToSend, 'r', encoding = 'utf8') as file: # open and read msgToSend txt file
        msgToEncode = file.read().replace('\n', '')       # remove possible break line  
        msgToEncode = msgToEncode[:80]                    # max 80 characters                          

    # Encode UTF8 character into binary sequence
    try:
        binaryMsg = ''.join((format(byte, '08b') for byte in msgToEncode.encode('utf-8', 'strict')))
    except UnicodeError:
        binaryMsg = "Message was not encoded in UTF-8"
        print("Message is not encoded in UTF-8")
    print("Message sent is : \n\'%s\'" % msgToEncode)
    print("Message sent is encoded into binary seq : \n\'%s\'" % binaryMsg)
    
    binaryMsg = zeroPaddingwithNull(binaryMsg) # zero padding to get a multiple of n greater than 640 bits
    binaryMsg = ControlBitCoding(binaryMsg)    # XOR Coding
    if coding_scheme == 'hadamard':
        binaryMsg = HadamardCoding(binaryMsg)  # Hadamard coding
    else:
        binaryMsg = RandomCoding(binaryMsg)    # Random coding

    # Map 1 -> 1\n and 0 -> -1\n
    x = binaryMsg.replace('1', '1\n')
    x = x.replace('0', '-1\n')

    with open('input.txt', "w") as text_file:
        text_file.write(x) # Write binary encoding of msgToEncode in input.txt
    return x
def RandomCoding(binaryMsg):
    binaryMsg = np.reshape(np.array(list(binaryMsg), dtype=int), (-1, n))
    binaryMsg = np.mod(np.dot(binaryMsg , G).flatten(), 2)
    binaryMsg = np.reshape(binaryMsg, (3, -1)).T  # Arrange bits s.t we recover bits even if 1/3 bits are ereased (periodically)  
    binaryMsg = arrayToStringAlongAxis(binaryMsg, 1)
    return arrayToString1D(binaryMsg)
def HadamardCoding(binaryMsg): # i.e orthogonal coding
    binaryMsg = np.reshape(np.array(list(binaryMsg), dtype=int), (-1, n))
    binaryMsg = np.mod(np.delete(np.dot(binaryMsg , G), 0, 1).flatten(), 2)
    binaryMsg = np.reshape(binaryMsg, (3, -1)).T  # Arrange bits s.t we recover bits even if 1/3 bits are ereased (periodically)  
    binaryMsg = arrayToStringAlongAxis(binaryMsg, 1)
    return arrayToString1D(binaryMsg)
def ControlBitCoding(binaryMsg): # i.e XOR coding
    return binaryMsg + arrayToString1D(('0' if c1 == c2 else '1') for c1, c2 in zip(binaryMsg[:int(len(binaryMsg)/2)], binaryMsg[int(len(binaryMsg)/2):]))
# -------------------------------------------------------------------------------------------------

# -------------------- Transmission channel : local or online --------------------
def transmission(server): # read input.txt, apply noise and save result in output.txt
    if server == "local":
        signal   = np.loadtxt('input.txt')
        N_sample = signal.size
        print("Number of sample used is n=%s" %N_sample)
        if not ((signal.shape == (N_sample,)) and
            np.issubdtype(signal.dtype, np.floating)):
            raise ValueError('signal must contain a real-valued sequence.')

        if N_sample > 60000:
            raise ValueError(('signal contains more than 32,000 samples. '
                        'Design a more efficient communication system.'))

        np.savetxt('output.txt', channel(signal))
    if server == "online": # Need to be connected to EPFL VPN
        os.system("python client.py --input_file input.txt --output_file output.txt --srv_hostname iscsrv72.epfl.ch --srv_port 80") # Command line to call client.py
        time.sleep(0.5)
    return 0
def channel(chanInput): # Noisy channel
    chanInput                               = np.clip(chanInput, -1, 1)
    erasedIndex                             = np.random.randint(3)
    chanInput[erasedIndex:len(chanInput):3] = 0
    print("Ereased index is " + str(erasedIndex))
    return chanInput + np.sqrt(10)*np.random.randn(len(chanInput))
# --------------------------------------------------------------------------------

# -------------------- Receiver apply decoding scheme on output.txt and save result in msgReceived (txt file) --------------------
def receiver(msgReceived, coding_scheme):
    y = np.loadtxt('output.txt')

    hat_i = np.argmin([np.sum(np.absolute(y[::3])), np.sum(np.absolute(y[1::3])), np.sum(np.absolute(y[2::3]))])
    print("Estimate of ereased index is %s" % hat_i)
 
    y = np.delete(y, slice(hat_i, None, 3))      # delete guessed ereased bits by channel 
    y = np.concatenate((y[::2],y[1::2]), 0)      # Re-arrange bits in correct order
    if coding_scheme == 'hadamard':
        y = HadamardDecoding(y)                  # Hadamard coding
    else:
        y = RandomDecoding(y)                    # Random coding
    msgDecodedBin = ControlBitDecoding(y, hat_i) # XOR Decoding
    
    # Some terminal/PC prints 0x0000 as a space
    msgDecodedBin = msgDecodedBin.replace("0000000000000000", "") # Remove null character
    # Remove possible 0x00 char at end of binary sequence resulting from padding (again some terminal/PC prints 0x00 as a space)
    # Here we use fact that UTF8 chars are constrained to 1 byte (by design), otherwise line below brings a bug if last char is in format 0x??00
    if msgDecodedBin[-8:] == "00000000":
        msgDecodedBin = msgDecodedBin[:-8]
    msgDecodedBin = msgDecodedBin[:640]                           # Constraint to 640 bits (by design)

    # Binary sequence to utf8
    msgDecodedBin = [msgDecodedBin[8*i:8*i+8] for i in range(0, int(len(msgDecodedBin)/8))]
    msgDecodedBin = bytes([int(x,2) for x in msgDecodedBin])
    charMsg = msgDecodedBin.decode('utf-8', errors = 'replace')
    print("Message received is : \n\'%s\'" % charMsg)

    with open(msgReceived, "w", encoding = 'utf8') as text_file:
        text_file.write(charMsg) # Write received character in msgReceived txt file
    return charMsg
def RandomDecoding(y): 
    msgDecoded = ""
    for i in range(0, int(len(y)/(N))):
        tmp         = y[(N)*i:(N)*(i+1)] @ ((np.where(RandomCode <= 0, -1., 1.)).T)
        tmp         = convertElemArrayToBinary(np.argmax(tmp))[0]
        tmp         = arrayToStringAlongAxis(tmp, 0)
        msgDecoded += str(tmp)
    return msgDecoded
def HadamardDecoding(y): 
    msgDecoded = ""
    for i in range(0, int(len(y)/(2**n -1))):
        tmp         = y[(2**n - 1)*i:(2**n - 1)*(i+1)] @ ((np.where(HadCode <= 0, -1., 1.)).T)
        tmp         = convertElemArrayToBinary(np.argmax(tmp))[0]
        tmp         = arrayToStringAlongAxis(tmp, 0)
        msgDecoded += str(tmp)
    return msgDecoded
def ControlBitDecoding(y, j):
    if j == 1:
        return y[:int(len(y)/2)] + arrayToString1D(('0' if c1 == c2 else '1') for c1, c2 in zip(y[:int(len(y)/2)], y[int(len(y)/2):]))
    if j == 0: 
        return arrayToString1D(('0' if c1 == c2 else '1') for c1, c2 in zip(y[:int(len(y)/2)], y[int(len(y)/2):])) + y[:int(len(y)/2)]
    return y
# --------------------------------------------------------------------------------

# ------------------- Communication through a noisy channel ----------------------
def communicate(msgToSend: str, msgReceived: str, server: str, coding_scheme = str) -> str:   
    print("-- Start binary encoding of msg --")
    transmitter(msgToSend, coding_scheme)
    print("-- End binary encoding of msg : --")

    print("-- Start transmission --")
    transmission(server)
    print("-- End transmission --")

    print("-- Start decoding --")
    msgDecoded = receiver(msgReceived, coding_scheme)
    print("-- End decoding --")
    return msgDecoded
# --------------------------------------------------------------------------------


# Coding :  n bits -> N bits
if __name__ == '__main__':
    version_cl = b'dUN'                                   # Always length-3 alphanumeric
    args = parse_args()  

    global n, G, HadCode, RandomCode
    n = args.msg_length                         
    N = args.block_length
    CodingScheme = args.coding_scheme
    B =  convertElemArrayToBinary(np.arange(0, 2**n))     # Matrice containing all binary sequences {0,1}^n of length n
    if CodingScheme == 'hadamard':
        G = B.T                                           # Hadamard generating function (transpose of B) used to encode n-length seq. to 2**n - 1 length seq.
        HadCode = np.array([(x @ G)%2 for x in B])        # All Hadamard codewords {0,1}^(2^n - 1)
        HadCode = np.delete(HadCode, 0, 1)                #
    if CodingScheme == 'random':
        G = np.random.randint(0, 2, (n, N))               # Random code generating function used to encode n-length seq. to N length seq.
        RandomCode = np.array([(x @ G)%2 for x in B])     # All Random codewords {0,1}^N
    
    print("We use %s coding scheme with message length %s and block length %s" % (CodingScheme, n, N))
    communicate(args.send_file, args.receive_file, args.server, CodingScheme) 

    # Print nbr of errors
    with open(args.send_file, 'r', encoding = 'utf8') as file:
        msgToSend = file.read().replace('\n', '')
    with open(args.receive_file, 'r', encoding = 'utf8') as file:
        msgReceived = file.read().replace('\n', '')
    nbrErrors = sum(1 for a, b in zip(msgToSend, msgReceived) if a != b)
    print("Number of errors is : %s" % str(nbrErrors))
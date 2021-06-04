# PDC Project

## Principles of Digital Communication (COM301) Project at EPFL - 2021.

**This project aims to develop a system capable of reliably transmitting text files (UTF-8 characters) over a noisy channel.**


You will find more informations about project and channel description on `Presentation_slides.ppt` file. 

How to use ? Write a message to send in `msg-to-send.txt` file and test our implementation by running `main.py`. Received and decoded message will be accessible in `msg-received.py`.

*Requirement : Python 3 and NumPy.*

`main.py` implements a **Hadamard** (orthogonal) coding scheme over XOR coding scheme *OR* implements a **Random** coding scheme over a XOR coding scheme.

By default script (i.e `python main.py`) implements a Hadamard code with order 2^9 (msg length is 9 bit and block length is 2^9) over a channel simulated in local.
 
Exemple of running python script : 

```
python main.py
``` 

To have more control on script behavior you can pass arguments to `main.py`.

Some examples : 

```
python main.py --send_file msg-to-send.txt --receive_file msg-received.txt --server local  --msg_length 8 --block_length 370 --coding_scheme random

python main.py --send_file msg-to-send.txt --receive_file msg-received.txt --server local  --msg_length 9 --block_length 512 --coding_scheme hadamard

python main.py --send_file msg-to-send.txt --receive_file msg-received.txt --server online  --msg_length 8 --block_length 370 --coding_scheme random

python main.py --send_file msg-to-send.txt --receive_file msg-received.txt --server online  --msg_length 9 --block_length 512 --coding_scheme hadamard
```

Arguments :

```
--send_file     : txt file with UTF-8 characters to transmit (max 80 bytes). Default is msg-to-send.txt.

--receive_file  : txt file with UTF-8 characters received. Default is msg-received.txt.

--server        : Channel server to use either 'local' or 'online' (if 'online" need to be connected at EPFL VPN). Default is local.

--msg_length    : Message lenght. Default is 9.

--block_length  : Block lenght. If coding_scheme is hadamard must be equal to 2**msg_length. Default is 512.

--coding_scheme : Coding scheme either 'random' or 'hadamard'. Default is hadamard.
```

By Gaiëtan RENAULT - Sophia ARTIOLI - Alexandre SANTANGELO

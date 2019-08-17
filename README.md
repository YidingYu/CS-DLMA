## A short introduction to CS-DLMA
CS-DLMA is a carrier sense multiple access (CSMA) protocol for heterogeneous wirless networking. The underpinning technique of CS-DLMA is deep Q-network (DQN). However, conventional DQN algorithms cannot fit in with the specicial issues in CSMA protocols design. We put forth a new DQN algorithm to address these issuses. 

## How to use the codes
The codes in the "agent+WiFi" simulate the coexistence of one CS-DLMA node with one WiFi node. 
- *run_DQN.py* is the main framework, and you can run this file to start the simulation. 
- *environment.py* simulates the interaction between the CS-DLMA node and the WiFi node. It is similar as OpenAI gym environment.
- *DQN_brain.py* builds the CS-DLMA node/agent. The new DQN algorithm is here. 

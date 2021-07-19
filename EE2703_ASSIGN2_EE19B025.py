'''
Assignment-2 Solution
ee19b025
J Antonson
Feb 24, 2021
'''

'''
# NETLIST FILE-1
.circuit
V1 GND 1 ac 1 0
R1 1 2 4.5e3
L1 2 3 80.96e-6
C1 3 GND 2.485e-12
L1 3 4 80.96e-6
R2 4 GND 4e3
.end
.ac V1 1e6

--------------------------------------------------------------

Current direction is assumed to be from the higher potential node.

The Circuit solution is as follows:

The voltage at node GND is 0j
The voltage at node 2 is (0.239+0.0234j)
The voltage at node 3 is (0.236-0.0061j)
The voltage at node 4 is (0.232-0.0356j)
The voltage at node 1 is (0.5+0j)

The current through source V1 is (5.8e-05-5.2e-06j)

The M matrix is :
          1+0j                     0+0j           0+0j                 0+0j             0+0j   0+0j
          0+0j  0.000222222-0.00196585j  0+0.00196585j                 0+0j  -0.000222222+0j   0+0j
0-1.56137e-05j            0+0.00196585j  0-0.00391608j        0+0.00196585j             0+0j   0+0j
   -0.00025+0j                     0+0j  0+0.00196585j  0.00025-0.00196585j             0+0j   0+0j
          0+0j          -0.000222222+0j           0+0j                 0+0j   0.000222222+0j  -1+0j
         -1+0j                     0+0j           0+0j                 0+0j             1+0j   0+0j

The b matrix is :
 0j 0j 0j 0j 0j (0.5+0j)
'''

#impoting the necessary libraries
import sys
import numpy as np
from numpy import cos,sin

#assigning constants
CIRCUIT = '.circuit'
END = '.end'
AC = '.ac' 
gnd = 'GND'
w = 0

# defining the classes
class elements():
    def __init__(self,line):
        self.line = line
        self.tokens = self.line.split()
        self.from_node = self.tokens[1]
        self.to_node = self.tokens[2]
        if len(self.tokens) == 5:   #check if the number of tokens is 5
            self.value = float(self.tokens[4])
        elif len(self.tokens) == 6: #check if the number of tokens is 6
            Vm = float(self.tokens[4])/2
            phase = float(self.tokens[5])
            self.value = complex(Vm*cos(phase),Vm*sin(phase))
        else:
            self.value = float(self.tokens[3])

# defining the function to make dictionaries 
def convert_to_dict(cir_def):   # This code will be used to make the circuit definition into a dictionary
    node_diction = {}
    nodes = [elements(line).from_node for line in cir_def] #get all the nodes from the circuit definition
    nodes.extend([elements(line).to_node for line in cir_def])
    index = 1
    nodes = list(set(nodes))
    for node in nodes:     #go through all the nodes
        if node == gnd :   #check if the selected node is Ground Node
            node_diction[node] = 0
        else:
            node_diction[node] = index
            index += 1
    return node_diction

#function to get the key, given the input value
def get_key(diction,value):
    for key in diction.keys():
        if diction[key] == value :
            return key

# function to make dictionary from the circuit definition
def make_dict(cir_def,element):
    volt_dict = {}  #making a voltage dictionary
    # Place the voltage names
    volt_names = [elements(line).tokens[0] for line in cir_def if elements(line).tokens[0][0].lower()== element]
    for ind,name in enumerate(volt_names):
        volt_dict[name] = ind
    return volt_dict

#function to make matrix for AC circuits
def mod_matrix(cir_def,w,node_key,diction,volt_dict,ind_dict,M,b,n,k):
    #get all the indexes and store them in a array from the circuit definition
    inds = [(i,j) for i in range(len(cir_def)) for j in range(len(cir_def[i].split())) if cir_def[i].split()[j] in diction.keys() if diction[cir_def[i].split()[j]] == node_key]
    #go through all the elements of inds array
    for ind in inds:
        element = elements(cir_def[ind[0]])
        element_name = cir_def[ind[0]].split()[0]
        if element_name[0] == 'R':  #check if the element is a resistor
            if ind[1] == 1:
                adj_key = diction[element.to_node]          #make the second item as the "to node"
                M[node_key,node_key] += 1/(element.value)   # put 1/R in the correct position of M matrix
                M[node_key,adj_key] -= 1/(element.value)    # put -1/R in the correct position of the M Matrix
                    
            if ind[1] == 2 :
                adj_key = diction[element.from_node]        #make the third item as the "from node"
                M[node_key,node_key] += 1/(element.value)   # put 1/R in the correct position of the M Matrix
                M[node_key,adj_key] -= 1/(element.value)    # put -1/R in the correct position of the M Matrix
                
        if element_name[0] == 'C' : #check if the element is a capacitor
            if ind[1]== 1:
                adj_key = diction[element.to_node]                              #make the second item as the "to node"
                M[node_key,node_key] += complex(0, 2*np.pi*w*(element.value))   # put (2*pi*w*C)j in the correct position of the M Matrix
                M[node_key,adj_key] -= complex(0, 2*np.pi*w*(element.value))    # put -(2*pi*w*C)j in the correct position of the M Matrix
            if ind[1] == 2 :
                adj_key = diction[element.from_node]                            #make the third item as the "from node"
                M[node_key,node_key] += complex(0, 2*np.pi*w*(element.value))   # put (2*pi*w*C)j in the correct position of the M Matrix
                M[node_key,adj_key] -= complex(0, 2*np.pi*w*(element.value))    # put -(2*pi*w*C)j in the correct position of the M Matrix

        if element_name[0] == 'L' : #check if the element is a inductor
            try:
                if ind[1]== 1:
                    adj_key = diction[element.to_node]                             #make the second item as the "to node"
                    M[node_key,node_key] -= complex(0,1/(2*np.pi*w*element.value)) # put -1/(2*pi*w*L)*j in the correct position of the M Matrix
                    M[node_key,adj_key] += complex(0,1/(2*np.pi*w*element.value))  # put 1/(2*pi*w*L)*j in the correct position of the M Matrix
                if ind[1] == 2 :
                    adj_key = diction[element.from_node]                             #make the third item as the "from node"
                    M[node_key,node_key] -= complex(0,1/(2*np.pi*w*element.value))   # put -1/(2*pi*w*L)*j in the correct position of the M Matrix
                    M[node_key,adj_key] += complex(0,1/(2*np.pi*w*element.value))    # put 1/(2*pi*w*L)*j in the correct position of the M Matrix
            #Check for zero division error
            except ZeroDivisionError:
                index = ind_dict[element_name]
                if ind[1]== 1:
                    adj_key = diction[element.to_node] #make the second item as the "to node"
                    M[node_key,n+k+index] += 1 
                    M[n+k+index,node_key] -= 1
                    b[n+k+index] = 0
                if ind[1]== 2:
                    M[node_key,n+k+index] -= 1
                    M[n+k+index,node_key] += 1
                    b[n+k+index] = 0
        if element_name[0] == 'V' : #check if the selected element is a Voltage source
            index = volt_dict[element_name]
            if ind[1]== 1:
                adj_key = diction[element.to_node]  #make the second item as the "to node"
                M[node_key,n+index] += 1        #assign +1 to the correct location of M matrix
                M[n+index,node_key] -= 1        #assign -1 to the correct location of M matrix
                b[n+index] = element.value      #adding V value in the b matrix at the correct position
            if ind[1] == 2 :
                adj_key = diction[element.from_node]    #make the third item as the "from node"
                M[node_key,n+index] -= 1        #assign -1 to the correct location of M matrix
                M[n+index,node_key] +=1         #assign +1 to the correct location of M matrix
                b[n+index] = element.value      #adding V value in the b matrix at the correct position

        if element_name[0] == 'I' : #check if the selected element is a Current source
            if ind[1]== 1:
                b[node_key] -= element.value    #adding -I value in the b matrix at the correct position
            if ind[1] == 2 :
                b[node_key] += element.value    #adding I value in the b matrix at the correct position

# function for neatly printing the matrix
def matprint(mat, fmt="g"):
    col_maxes = [max([len(("{:"+fmt+"}").format(x)) for x in col]) for col in mat.T]
    for x in mat:
        for i, y in enumerate(x):
            print(("{:"+str(col_maxes[i])+fmt+"}").format(y), end="  ")
        print("")


'''
Execution of the main function starts here!
'''

# using try-catch
try:
    if( len(sys.argv) != 2): #check if the input has two arguments
        sys.exit('Invalid number of arguments')
    name = sys.argv[1]
    file = open(name)
    lines = file.readlines()
    file.close()

    # Check if the given circuit has AC definition
    for line in lines:
        if line[:len(AC)] == AC :
            w = float(line.split()[2])  
        
    # getting the circuit definition block :
    for ind,string in enumerate(lines):
        try : 
            if string.split()[0] == CIRCUIT:
                start_ind = ind
            elif string.split()[0] == END:
                end_ind = ind
        except IndexError: 
            continue

    #checking of the file format is correct
    try:
        cir_def = lines[start_ind+1:end_ind]
    except NameError:
        print("\nInvalid file format!\n")
        exit()

    # removing the comments from circuit definition
    mod_def = []
    for line in cir_def:
        index = len(line)-1
        for i in range(len(line)):
            if line[i] == '#':
                index = i
                break
        mod_def.append(line[:index])
    cir_def = mod_def

    # Make the dictionaries
    diction = convert_to_dict(cir_def)
    volt_dict = make_dict(cir_def,'v')
    ind_dict = make_dict(cir_def,'l')

    # to find the dimension of the M matrix
    k =0
    for i in range(len(cir_def)):
        if cir_def[i].split()[0][0] == 'V':
            k+=1
    n = len(convert_to_dict(cir_def))

    # Find the dimensions
    dim = n+k

    # Creating zero matrix in desired size (M and b)
    M = np.zeros((dim,dim),dtype=np.complex)
    b = np.zeros(dim,dtype=np.complex)

    dc_flag = False

    # if the circuit is DC, then do this
    if w == 0:
        dc_flag = True
        M = np.zeros((dim+len(ind_dict),dim+len(ind_dict)),dtype=np.complex)    # make a complex zero matrix M
        b = np.zeros(dim+len(ind_dict),dtype=np.complex)                        # make a complex zero matrix b

    for i in range(len(diction)):
        mod_matrix(cir_def,w,i,diction,volt_dict,ind_dict,M,b,n,k)              # Make the modified matrix

    # for reference voltage
    M[0] = 0
    M[0,0] =1

    # print the M and b matrix
    print('\nThe M matrix is :'); matprint(M)
    print('\nThe b matrix is :\n',*b)

    # For printing and checking the final outputs
    try:
        x = np.linalg.solve(M,b)   
        print('\nCurrent direction is assumed to be from the higher potential node.\n') 
    except Exception:
        print('\nInvalid Circuit, Solution does not exist!\n')
        sys.exit()

    print("The Circuit solution is as follows: \n")
    for i in range(n):
        print("The voltage at node {} is {:.3}".format(get_key(diction,i),x[i]))
    print('\n')
    for j in range(k):
        print('The current through source {} is {:.3}'.format(get_key(volt_dict,j),x[n+j]))
    print('\n')
    if dc_flag:
        for i in range(len(ind_dict)):
            print("The current through inductor {} is {:.3}".format(get_key(ind_dict,i),x[n+k+i]))

# if there is a error that was caught, then the file format is wrong!  
except Exception:
   print('\nInvalid file!\n')

'''
# NETLIST FILE-2
.circuit
V1 GND n1 ac 1 0
R1 n1 n2 4.5e3
L1 n2 n3 80.96e-6
L2 n3 n4 80.96e-6
C1 GND n3 2.485e-12
R2 GND n4 4e3
.end
.ac V1 1000

--------------------------------------------------------------

The M matrix is :
          1+0j             0+0j              0+0j                  0+0j        0+0j   0+0j
          0+0j   0.000222222+0j              0+0j       -0.000222222+0j        0+0j  -1+0j
   -0.00025+0j             0+0j  0.00025-1.96585j                  0+0j  0+1.96585j   0+0j
          0+0j  -0.000222222+0j              0+0j  0.000222222-1.96585j  0+1.96585j   0+0j
0-1.56137e-08j             0+0j        0+1.96585j            0+1.96585j  0-3.93169j   0+0j
         -1+0j             1+0j              0+0j                  0+0j        0+0j   0+0j

The b matrix is :
 0j 0j 0j 0j 0j (0.5+0j)

Current direction is assumed to be from the higher potential node.

The Circuit solution is as follows:

The voltage at node GND is 0j
The voltage at node n1 is (0.5+0j)
The voltage at node n4 is (0.235-3.59e-05j)
The voltage at node n2 is (0.235+2.39e-05j)
The voltage at node n3 is (0.235-6.02e-06j)


The current through source V1 is (5.88e-05-5.31e-09j)
'''

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  4 17:43:38 2021

@author: p0g02rj
"""
import pandas as pd
import networkx as nx # for drawing graphs
import matplotlib.pyplot as plt # for drawing graphs

# for creating Bayesian Belief Networks (BBN)
from pybbn.graph.dag import Bbn
from pybbn.graph.edge import Edge, EdgeType
from pybbn.graph.jointree import EvidenceBuilder
from pybbn.graph.node import BbnNode
from pybbn.graph.variable import Variable
from pybbn.pptc.inferencecontroller import InferenceController


	# Set Pandas options to display more columns
pd.options.display.max_columns=50

# Read in the weather data csv
df=pd.read_csv('./WheatClassificationData.csv', encoding='utf-8')
#df1=pd.read_csv('./weatherAUS.csv', encoding='utf-8')
corr = df.corr(method='pearson')
corr.head()
def condProbTable(a , given , data):
    if len(given)!=0:
        Given=list(given)
        Given.append(a)
        Probs=data.groupby(Given).size()/data.groupby(given).size()
        A = pd.DataFrame(Probs)
        Indexes = A.index
        C=pd.DataFrame()
        print(C)
        FirstColName='('
        for i in range(0,len( Indexes.names )-1):
            FirstColName=FirstColName+Indexes.names[i]+' '
            if(i==len(Indexes.names)-2):
                FirstColName=FirstColName+')'
            else:
                FirstColName=FirstColName+','
        C[FirstColName]=[Indexes[i][0:len(given)] for i in range(0,len(Indexes))]
        D=C.copy()
        C=C.drop_duplicates().reset_index(drop =True)
        aValues= [Indexes[i][len(given)] for i in range(0,len(Indexes))]
        for i in list(set(aValues)):
            C[a+' = '+str(i)] = 0.
        for i in range(0,len(C)):
            for j in D[D.iloc[:,0]==C.iloc[:,0][i]].index.values:
                x=list(C.iloc[i,0])
                x.append(aValues[j])
                x=tuple(x)
                C[a+' = '+str(aValues[j])][i]=A.loc[x]
    else:
        Probs=data.groupby(a).size()/len(data)
        C = pd.DataFrame(Probs)
        C.columns=['Probability']
        C.reset_index()
    return C




"""
g = df1.mean()
h = df1.median()
i = df1.mode()
"""
"""
Q1 = df['Compactness'].quantile(0.25)
Q2 = df['Compactness'].quantile(0.50)
Q3 = df['Compactness'].quantile(0.75)
IQR = Q3 - Q1
print(IQR)
plotdata = df['Compactness']
plotdata.plot(kind="bar")
"""
df['Length of wheat']=df['Length of kernel'].apply(lambda x: '0.<=5.262'   if x<=5.262 else
                                                            '5.262-5.527' if 5.262<x<=5.527 else'5.527>')
df['Breadth of wheat']=df['Width of kernel'].apply(lambda x: '0.<=2.947'   if x<=2.947 else
                                                            '2.947-3.242' if 2.947<x<=3.242 else'3.242>')
df['Compactness of wheat']=df['Compactness'].apply(lambda x: '0.<=0.8571'   if x<=0.8571 else
                                                            '0.8571-0.8735' if 0.8571<x<=0.8735 else'0.8735>')

df['Asymmetry coefficient of wheat']=df['Asymmetry coefficient'].apply(lambda x: '0.<=2.570'   if x<=2.57 else
                                                            '2.570-3.598' if 2.57<x<=3.598 else'3.598>')
df['Area of wheat']=df['Area'].apply(lambda x: '0.<=12.280'   if x<=12.280 else
                                                            '12.280-14.27' if 12.280<x<=14.37 else'14.37>')

Length_of_wheat = condProbTable('Length of wheat',given= [], data=df)
Breadth_of_wheat = condProbTable('Breadth of wheat', given= [], data=df)
Compactness_of_wheat = condProbTable('Compactness of wheat', given= [], data=df)
Asymmetry_coefficient_of_wheat= condProbTable('Asymmetry coefficient of wheat', given= [], data=df)
Area_of_wheat= condProbTable('Area of wheat', given= ['Length of wheat','Breadth of wheat'], data=df)
wheat_Type= condProbTable('Wheat variety', given= ['Area of wheat','Asymmetry coefficient of wheat', 'Compactness of wheat'], data=df)

"""
Manually puttingthe frequence better way to do is to customize it
"""
"""
wheatLength = BbnNode(Variable(0, 'wheatLength', ['<=5.262','5.262-5.527','>5.527']), [0.251185,0.251185, 0.49763])

wheatBreadth= BbnNode(Variable(1, 'wheatBreadth', ['<=2.947','2.947-3.242','>3.242']), [0.251185,0.251185, 0.49763])
areaWheat= BbnNode(Variable(2, 'areaWheat', ['<=12.28','12.28-14.27','>14.27']), [0, 0, 1,
                                                0, 0.75, 0.25,
                                                0.50, 0.50, 1,
                                                0, 0.2272, 0.7727,
                                                0.6363, 0.3636, 0,
                                                0.3846, 0.6153, 1,
                                                1, 0, 0,])

#wheatCompactness= BbnNode(Variable(3 'wheatCompactness', ['<=0.8879','>0.8879']),[0.7488,0.251185])
wheatAssymtric= BbnNode(Variable(4, 'wheatAssymtric', ['<=2.570','2.570-3.598','>3.598']), [0.251185,0.251185, 0.49763])
wheatType= BbnNode(Variable(5, 'wheatType', ['1','2','3']), [0.5, 0, 0.5,
                                                0.166, 0, 0.83,
                                                0, 0, 1,
                                                1, 0, 0,
                                                0.666, 0, 0.333,
                                                0.769, 0, 0.230,
                                                0.75, 0, 0.25,
                                                0.166, 0, 0.833,
                                                0.75, 0, 0.25,
                                                0.578, 0.421, 0,
                                                0.375, 0.625, 0,
                                                0.214, 0.714, 0.071,
                                                0.1428, 0.857, 0,])
"""
def probs(data, child, parent1=None, parent2=None):
    if parent1==None:
        # Calculate probabilities
        prob=pd.crosstab(data[child], 'Empty', margins=False, normalize='columns').sort_index().to_numpy().reshape(-1).tolist()
    elif parent1!=None:
            # Check if child node has 1 parent or 2 parents
            if parent2==None:
                # Caclucate probabilities
                prob=pd.crosstab(data[parent1],data[child], margins=False, normalize='index').sort_index().to_numpy().reshape(-1).tolist()
            else:
                # Caclucate probabilities

                prob=pd.crosstab([data[parent1],data[parent2]],data[child], margins=False, normalize='index').sort_index().to_numpy().reshape(-1).tolist()
    else: print("Error in Probability Frequency Calculations")
    print(prob)
    return prob


wheatLength = BbnNode(Variable(0, 'length', ['<=5.262','5.262-5.527','>5.527']), probs(df, child = 'Length of wheat'))

wheatBreadth= BbnNode(Variable(1, 'breadth', ['<=2.947','2.947-3.242','>3.242']), probs(df, child = 'Breadth of wheat'))

wheatCompactness= BbnNode(Variable(2, 'compact', ['<=0.8571','0.8571-0.8879','>0.8879']), probs(df, child = 'Compactness of wheat'))

wheatAssymtric= BbnNode(Variable(3, 'assymtric', ['<=2.570','2.570-3.598','>3.598']), probs(df, child = 'Asymmetry coefficient of wheat'))


wheatArea= BbnNode(Variable(4, 'area',['0.<=12.280', '=14.37>', '12.280 - 14.27']), probs(df, child = 'Area of wheat', parent1 = 'Length of wheat', parent2 = 'Breadth of wheat'))



wheatType= BbnNode(Variable(5, 'type', ['1','2','3']), [0.5, 0, 0.5,
                                                0.166, 0, 0.83,
                                                0, 0, 1,
                                                1, 0, 0,
                                                0.666, 0, 0.333,
                                                0.769, 0, 0.230,
                                                0.75, 0, 0.25,
                                                0.166, 0, 0.833,
                                                0.75, 0, 0.25,
                                                0.578, 0.421, 0,
                                                0.571, 0.428, 0,
                                                0.375, 0.625, 0,
                                                0.214, 0.714, 0.071,
                                                0.1428, 0.857, 0,
                                                0.1428, 0.857, 0,])


bbn = Bbn() \
    .add_node(wheatLength) \
    .add_node(wheatBreadth) \
    .add_node(wheatCompactness) \
    .add_node(wheatAssymtric) \
    .add_node(wheatArea) \
    .add_node(wheatType) \
    .add_edge(Edge(wheatLength, wheatArea, EdgeType.DIRECTED)) \
    .add_edge(Edge(wheatBreadth, wheatArea, EdgeType.DIRECTED)) \
    .add_edge(Edge(wheatCompactness, wheatType, EdgeType.DIRECTED)) \
    .add_edge(Edge(wheatAssymtric, wheatType, EdgeType.DIRECTED))\
    .add_edge(Edge(wheatArea, wheatType, EdgeType.DIRECTED))

# Convert the BBN to a join tree



# Set node positions
pos = {0: (-3, 3), 1: (3, 3), 2: (3, 0), 3: (-2, 1), 4: (1.5,1.5), 5: (-1, -1)}

# Set options for graph looks
options = {
    "font_size": 16,
    "node_size": 4000,
    "node_color": "white",
    "edgecolors": "black",
    "edge_color": "red",
    "linewidths": 5,
    "width": 5,}

# Generate graph
n, d = bbn.to_nx_graph()
nx.draw(n, with_labels=True, labels=d, pos=pos, **options)

# Update margins and print the graph
ax = plt.gca()
ax.margins(0.10)
plt.axis("off")
plt.show()

join_tree = InferenceController.apply(bbn)
"""
bbn1 = Bbn() \
    .add_node(wheatArea) \
    .add_edge(Edge(wheatArea, wheatType, EdgeType.DIRECTED))

"""



# Define a function for printing marginal probabilities
def print_probs():
    for node in join_tree.get_bbn_nodes():
        potential = join_tree.get_bbn_potential(node)
        print("Node:", node)
        print("Values:")
        print(potential)
        print('----------------')

# Use the above function to print marginal probabilities
print_probs()


def evidence(ev, nod, cat, val):
    ev = EvidenceBuilder() \
    .with_node(join_tree.get_bbn_node_by_name(nod)) \
    .with_evidence(cat, val) \
    .build()
    join_tree.set_observation(ev)

# Use above function to add evidence
evidence('ev1', 'area', '0.<=12.280', 1.0)
evidence('ev1', 'compact', '>0.8879', 1.0)
#evidence('ev1', 'area', '0.<=12.280', 1.0)
# Print marginal probabilities
print_probs()







"""
Length of wheat = condProbTable('Length of wheat', given= [], data=df)
Breadth of wheat = condProbTable('Breadth of wheat', given= [], data=df)
Compactness of wheat = condProbTable('Compactness of wheat', given= [], data=df)
Asymmetry coefficient of wheat= condProbTable('Asymmetry coefficient of wheat', given= [], data=df)
Area of wheat= condProbTable('Area of wheat', given= ['Length of wheat','Breadth of wheat'], data=df)
"""
"""
#filtered = df.query('(@Q1 - 1.5 * @IQR) <= Length of kernel <= (@Q3 + 1.5 * @IQR)')

print("\n----------- Calculate Mean -----------\n")
print(df.mean())

print("\n----------- Calculate Median -----------\n")
print(df.median())

print("\n----------- Calculate Mode -----------\n")
print(df.mode())
"""



"""
# Create bands for variables that we want to use in the model
df['WindGustSpeedCat']=df['WindGustSpeed'].apply(lambda x: '0.<=40'   if x<=40 else
                                                            '1.40-50' if 40<x<=50 else '2.>50')
df['Humidity9amCat']=df['Humidity9am'].apply(lambda x: '1.>60' if x>60 else '0.<=60')
df['Humidity3pmCat']=df['Humidity3pm'].apply(lambda x: '1.>60' if x>60 else '0.<=60')

# Show a snaphsot of data
df

H9am = BbnNode(Variable(0, 'H9am', ['<=60', '>60']), [0.30658, 0.69342])
H3pm = BbnNode(Variable(1, 'H3pm', ['<=60', '>60']), [0.92827, 0.07173,
                                                      0.55760, 0.44240])
W = BbnNode(Variable(2, 'W', ['<=40', '40-50', '>50']), [0.58660, 0.24040, 0.17300])
RT = BbnNode(Variable(3, 'RT', ['No', 'Yes']), [0.92314, 0.07686,
                                                0.89072, 0.10928,
                                                0.76008, 0.23992,
                                                0.64250, 0.35750,
                                                0.49168, 0.50832,
                                                0.32182, 0.67818])
"""

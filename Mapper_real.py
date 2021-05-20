import kmapper as km
import numpy as np
import matplotlib.pyplot as plt
import ot
import sys, os
import array_to_latex as a2l
from kmapper.plotlyviz import plotlyviz 
import plotly.graph_objects as go
import networkx as nx
from sklearn import preprocessing

np.random.seed(16)

def blockPrint():
    sys.stdout = open(os.devnull, 'w')

def enablePrint():
    sys.stdout = sys.__stdout__


# the function below calculates the average filter values for each node

def avg_filter(node,mapper,graph,my_data):
    sum = 0
    data_from_node = mapper.data_from_cluster_id(node,graph,my_data)
    for y in range(len(data_from_node)):
        sum = sum + (mapper.fit_transform(data_from_node[y:y+1], projection = [1], scaler = None))[0][0]
    return sum/len(data_from_node)
   

# code below should calculate the measure


def calc_measure(graph, mapper, my_data):

    measure = np.ndarray(shape = (1,len(graph["nodes"])))
    sum = 0
    itera = 0
    
    for x in graph["nodes"]:
        sum = sum + len(mapper.data_from_cluster_id(x,graph,my_data))

    for x in graph["nodes"]:
        measure[0][itera] = len(mapper.data_from_cluster_id(x,graph,my_data))/sum
        itera = itera+1
        
    return measure



# we will do Dijkstra's below to get the path induced distance matrix (want to get the shortest path
#  between any two nodes)

# following code is from geeks for geeks
# see https://www.geeksforgeeks.org/dijkstras-shortest-path-algorithm-greedy-algo-7/


import sys 
   
class Graph(): 

      def __init__(self, vertices): 
         self.V = vertices 
         self.graph = [[0 for column in range(vertices)]  
                    for row in range(vertices)] 

      def printSolution(self, dist): 
           print ("Vertex tDistance from Source") 
           for node in range(self.V): 
               print (node, "t", dist[node]) 

      def minDistance(self, dist, sptSet): 

            min = sys.maxsize
            min_index = 0

            for v in range(self.V): 
                if dist[v] < min and sptSet[v] == False: 
                    min = dist[v] 
                    min_index = v 
       
            return min_index 

      def induce_path_distmat(self,dist):
          IP_mat = np.zeros(self.V)
          for node in range(self.V):
               IP_mat[node] = dist[node]
          return IP_mat

      def dijkstra(self, src):

          dist = [sys.maxsize] * self.V 
          dist[src] = 0
          sptSet = [False] * self.V

          for cout in range(self.V): 

              u = self.minDistance(dist, sptSet)

              sptSet[u] = True

              for v in range(self.V): 
                  if self.graph[u][v] > 0 and sptSet[v] == False and dist[v] > dist[u] + self.graph[u][v]: 
                           dist[v] = dist[u] + self.graph[u][v]

          # self.printSolution(dist)
          return self.induce_path_distmat(dist)


def path_matrix(graph,mapper,my_data):
    nerve = km.GraphNerve()
    (edges,simplices) = nerve.compute(graph["nodes"])
    
    my_list = []
    
    for x in graph["nodes"]:
        my_list.append(x)

    D = np.zeros((len(graph["nodes"]),len(graph["nodes"])))

    for i in range(len(my_list)):
        for y in edges:
            if my_list[i] == y:
                   for j in range(len(edges[y])):
                         value = abs(avg_filter(edges[y][j],mapper,graph,my_data) - 
                                 avg_filter(my_list[i],mapper,graph,my_data))
                         p = my_list.index(edges[y][j])
                         D[i][p] = value
                         D[p][i] = value


    g = Graph(len(graph["nodes"]))
    g.graph = D
    M = np.zeros((len(graph["nodes"]),len(graph["nodes"])))

    for i in range(len(graph["nodes"])):
        M[i] = g.dijkstra(i)


    M = np.where(M == sys.maxsize, -1, M)
    max_value = np.max(M)
    M = np.where(M == -1, max_value, M)

    return M



def calc_gromov(graph,mapper,my_data,graph2,mapper2,my_data2):

    blockPrint()
    
    M1 = path_matrix(graph,mapper,my_data)
    M2 = path_matrix(graph2,mapper2,my_data2)

    meas1 = calc_measure(graph,mapper,my_data)
    meas2 = calc_measure(graph2,mapper2,my_data2)

    meas1 = np.squeeze(meas1)
    meas2 = np.squeeze(meas2)

    gw, log0 = ot.gromov.gromov_wasserstein2(M1,M2,meas1,meas2,'square_loss', verbose = True, log = True)


    #print('Gromov distance:')
    #print(gw)

    return gw   


def GW_matrix(num_circ,num_line,num_y,num_eight):
   
    total_num = num_circ + num_line + num_y + num_eight

    class Shape:
          def __init__(self,shape_type,graph,data,mapper):
              self.shape = shape_type
              self.graph = graph
              self.data = data
              self.mapper = mapper

    shapes = []

   
    blockPrint()

    points_in_interval = np.arange(0,1,0.0065)
    points_in_line = np.arange(-2,2,0.026)

    X = [2*np.cos(t*2*np.pi) for t in points_in_interval]
    Y = [2*np.sin(t*2*np.pi) for t in points_in_interval]

    Z = [t for t in points_in_line]
    W = [t for t in points_in_line]

    points_in2 = np.arange(-2.0,0.1,0.04)
    points_in3 = np.arange(0.1,2.0,0.04)
    P = [t for t in points_in2]
    a = [t for t in points_in3]
    a = np.repeat(a, 2, axis=0)
    P = np.concatenate((P,a))

    G = [0 for t in points_in2]
    V = [(t-0.1) for t in points_in3]
    V = np.repeat(V, 2, axis=0)
    V[::2] = -V[::2]
    G = np.concatenate((G,V))

    t = np.linspace(0, 2 * np.pi, 150)
    q = 2*(np.sin(t))
    m = 2*(np.sin(t) * np.cos(t))

    #scaler = 0

    for i in range(num_circ):         

         X_noise = np.random.normal(scale=0.1, size=len(X))
         Y_noise = np.random.normal(scale=0.1, size=len(Y))
         X_noise = np.array(X) + np.array(X_noise)
         Y_noise = np.array(Y) + np.array(Y_noise)

         plt.figure()
         plt.scatter(X_noise, Y_noise)
         plt.axis('equal')


         circ_data = np.array(list(zip(X_noise,Y_noise)))
         circ_mapper = km.KeplerMapper(verbose=1)
         circ_proj_data = circ_mapper.fit_transform(circ_data, projection = [1], scaler = None)
         circ_graph = circ_mapper.map(circ_proj_data, circ_data, cover=km.Cover(n_cubes= 6,
                      perc_overlap=0.3))
         circ_mapper.visualize(circ_graph, path_html= str(i) + "circle_keppler.html",
                         title="mapper applied to circle" + str(i))

         circ_obj = Shape("circle",circ_graph,circ_data,circ_mapper)

         shapes.append(circ_obj)

         #scaler = scaler + 0.35/num_circ

    for i in range(num_line):
         Z_noise = np.random.normal(scale=0.1, size=len(Z))
         W_noise = np.random.normal(scale=0.1, size=len(W))
         Z_noise = np.array(Z) + np.array(Z_noise)
         W_noise = np.array(W) + np.array(W_noise)


         plt.figure()
         plt.scatter(W_noise, Z_noise)
         plt.axis('equal')


         my_data_line = np.array(list(zip(W_noise,Z_noise)))

         mapper_line = km.KeplerMapper(verbose = 1)
         projected_data_line = mapper_line.fit_transform(my_data_line, projection = [1], scaler = None)

         graph_line = mapper_line.map(projected_data_line, my_data_line, cover = km.Cover(n_cubes = 6,
                      perc_overlap = .3))

         mapper_line.visualize(graph_line, path_html= str(i) + "line_keppler.html",
                    title="mapper applied to line" + str(i))

         line_obj = Shape("line",graph_line,my_data_line,mapper_line)

         shapes.append(line_obj)

            
     
    for i in range(num_y): 
         G_noise = np.random.normal(scale=0.1, size=len(G))
         P_noise = np.random.normal(scale=0.1, size=len(P))
         G_noise = np.array(G) + np.array(G_noise)
         P_noise = np.array(P) + np.array(P_noise) 

         plt.figure()
         plt.scatter(G_noise, P_noise)
         plt.axis('equal')


         y_data = np.array(list(zip(G_noise,P_noise)))
         y_mapper = km.KeplerMapper(verbose=1)
         y_proj_data = y_mapper.fit_transform(y_data, projection = [1], scaler = None)
         y_graph = y_mapper.map(y_proj_data, y_data, cover=km.Cover(n_cubes=6,
                      perc_overlap=0.3))
         y_mapper.visualize(y_graph, path_html = str(i) + "y_keppler.html",
                         title="mapper applied to y-shape" + str(i))

         y_obj = Shape("y",y_graph,y_data,y_mapper)

         shapes.append(y_obj)
         

    for i in range(num_eight):
         m_noise = np.random.normal(scale=0.1, size=len(m))
         q_noise = np.random.normal(scale=0.1, size=len(q))
         m_noise = np.array(m) + np.array(m_noise)
         q_noise = np.array(q) + np.array(q_noise)

         plt.figure()
         plt.scatter(m_noise, q_noise)
         plt.axis('equal')

 
         eight_data = np.array(list(zip(m_noise,q_noise)))
         eight_mapper = km.KeplerMapper(verbose=1)
         eight_proj_data = eight_mapper.fit_transform(eight_data, projection = [1], scaler = None)
         eight_graph = eight_mapper.map(eight_proj_data, eight_data, cover=km.Cover(n_cubes = 6,
                      perc_overlap=0.3))
         eight_mapper.visualize(eight_graph, path_html = str(i) + "eight_keppler.html",
                         title="mapper applied to figure 8" + str(i))

         eight_obj = Shape("figure 8",eight_graph,eight_data,eight_mapper)

         shapes.append(eight_obj)

    enablePrint()

    shmat = np.ones((total_num, total_num))

    for i in range(total_num):
        for j in range(total_num):
            if i == j:
               shmat[i][j] = 0

            else:   
                shmat[i][j] = calc_gromov(shapes[i].graph,shapes[i].mapper,shapes[i].data,
                                      shapes[j].graph,shapes[j].mapper,shapes[j].data)


    print(shmat)
    print(' ')
    a2l.to_ltx(shmat, frmt = '{:6.2f}', arraytype = 'array')
    print(' ')
    print(shmat[0:4,0:4])

    triang(shmat)


    plt.figure()     
    im = plt.imshow(shmat, cmap = 'hot')
    plt.colorbar(im)
    plt.show()

# we calculate gromov array with this function call



def real_data():

   blockPrint()

   class Real:
          def __init__(self,graph,data,mapper):
              self.graph = graph
              self.data = data
              self.mapper = mapper

   realList = []


   f = open('chicago_election_data.csv','r')

   from numpy import genfromtxt

   my_data = genfromtxt('chicago_election_data.csv', dtype = float ,delimiter = ','
   , skip_header  = 1, usecols = [*range(1,51)])


   eman_mapper = km.KeplerMapper(verbose=1)

   eman_proj_data = eman_mapper.fit_transform(my_data, projection = [46], scaler = None)

   graph_eman = eman_mapper.map(eman_proj_data, my_data, cover = km.Cover(n_cubes = 40,
                      perc_overlap = .3))
   eman_mapper.visualize(graph_eman, path_html = "eman_keppler.html",
                    title="mapper applied to eman")

   eman_obj = Real(graph_eman,my_data,eman_mapper)

   realList.append(eman_obj)



   garcia_mapper = km.KeplerMapper(verbose=1)

   garcia_proj_data = garcia_mapper.fit_transform(my_data, projection = [49], scaler = None)

   graph_garcia = garcia_mapper.map(garcia_proj_data, my_data, cover = km.Cover(n_cubes = 40,
                      perc_overlap = .3))

   garcia_mapper.visualize(graph_garcia, color_values = my_data[:,49], path_html = "garcia_keppler.html", 
                 title="mapper applied to garcia")


 
   stuff = plotlyviz(graph_garcia, color_values = my_data[:,49])

   stuff.show()

   plt.figure()
   img = plt.imshow(np.array([[0,1]]), cmap="viridis")
   img.set_visible(False)
   plt.colorbar(orientation="vertical")
   plt.show() 

   garcia_obj = Real(graph_garcia,my_data,garcia_mapper)

   realList.append(garcia_obj)



   lhg_mapper = km.KeplerMapper(verbose=1)

   lhg_proj_data = lhg_mapper.fit_transform(my_data, projection = [43], scaler = None)

   graph_lhg = lhg_mapper.map(lhg_proj_data, my_data, cover = km.Cover(n_cubes = 40,
                      perc_overlap = .3))

   lhg_mapper.visualize(graph_lhg, path_html = "lhg_keppler.html",
                    title="mapper applied to lhg")

   lhg_obj = Real(graph_lhg,my_data,lhg_mapper)

   realList.append(lhg_obj)



   preck_mapper = km.KeplerMapper(verbose=1)

   preck_proj_data = preck_mapper.fit_transform(my_data, projection = [35], scaler = None)

   graph_preck = preck_mapper.map(preck_proj_data, my_data, cover = km.Cover(n_cubes = 40,
                      perc_overlap = .3))

   preck_mapper.visualize(graph_preck, path_html = "preck_keppler.html",
                    title="mapper applied to preck")

   preck_obj = Real(graph_preck,my_data,preck_mapper)

   realList.append(preck_obj)


   enablePrint()

   rmat = np.ones((4, 4))

   for i in range(4):
       for j in range(4):
           if i == j:
              rmat[i][j] = 0

           else:
                rmat[i][j] = calc_gromov(realList[i].graph,realList[i].mapper,realList[i].data,
                                      realList[j].graph,realList[j].mapper,realList[j].data)


   print(rmat)
   print(' ')
   a2l.to_ltx(rmat, frmt = '{:6.2f}', arraytype = 'array')
   print(' ')

   #plt.figure()    
   #im = plt.imshow(rmat, cmap = 'hot')
   #plt.colorbar(im)
   #plt.show()




def real_noise():

   blockPrint()

   class Real:
          def __init__(self,graph,data,mapper):
              self.graph = graph
              self.data = data
              self.mapper = mapper

   realList = []


   f = open('chicago_election_data_modified.csv','r')

   from numpy import genfromtxt

   my_data = genfromtxt('chicago_election_data_modified.csv', dtype = float ,delimiter = ','
   , skip_header  = 1, usecols = [*range(1,33)])


   scaler2 = 0


   for i in range(10):
       

       data_noise = np.random.normal(scale=scaler2, size=my_data.shape)        

       data_noise = my_data + data_noise


       garcia_mapper = km.KeplerMapper(verbose=1)

       garcia_proj_data = garcia_mapper.fit_transform(data_noise, projection = [31], scaler = None)

       graph_garcia = garcia_mapper.map(garcia_proj_data, data_noise, cover = km.Cover(n_cubes = 40,
                      perc_overlap = .3))

       garcia_mapper.visualize(graph_garcia, color_values = garcia_proj_data, path_html = str(i) + "garcia_keppler.html"
                              ,title="mapper applied to garcia" + str(i)) 


       garcia_obj = Real(graph_garcia,data_noise,garcia_mapper)

       realList.append(garcia_obj)

       scaler2 = scaler2 + 0.1/10


      
   enablePrint()

   rmat = np.ones((10, 10))

   for i in range(10):
       for j in range(10):
           if i == j:
              rmat[i][j] = 0

           else:
                rmat[i][j] = calc_gromov(realList[i].graph,realList[i].mapper,realList[i].data,
                                      realList[j].graph,realList[j].mapper,realList[j].data)


   print(rmat)
   print(' ')
   a2l.to_ltx(rmat, frmt = '{:6.2f}', arraytype = 'array')
   print(' ')

   plt.figure()
   im = plt.imshow(rmat, cmap = 'hot')
   plt.colorbar(im)
   plt.show()
   



def triang(D):

  for i in range(10):
     for j in range(i):
        if any(D[i,j] > D[i,:] + D[:,j]):
           print("{},{}".format(i,j))



#GW_matrix(4,3,2,1)

#real_data()


real_noise()








     













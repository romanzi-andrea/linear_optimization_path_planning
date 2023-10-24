from robomarkt_1 import Cx, Cy, usable, Dc, maxdist, mindist, maxstores, Vc, Fc
import mip
import numpy as np
import math
import matplotlib.pyplot as plt

# _______________________________________________________________
# funzione per plottare una rappresentazione grafica del problema
def problem_plot(Cx, Cy, usable, x, max_circle=False, min_circle=False, links=False):
    
    points = []

    for i in range(len(Cx)):
        points.append((Cx[i],Cy[i],usable[i])) # crea una lista di tuple con le coordinate e se è usabile 

    points = np.array(points)

    figure, axes = plt.subplots()
    for i in range(len(Cx)):
        if points[i,2]==True:
            if x[i].x == True:
                plt.scatter(points[i,0], points[i,1], c='blue')
                plt.text(points[i,0], points[i,1], f'{i}') 
            else:
                plt.scatter(points[i,0],points[i,1],c='green')
        else:
            plt.scatter(points[i,0],points[i,1],c='red')
        plt.scatter(points[0,0],points[0,1],c='yellow')
    for i in range(len(Cx)):
        if x[i].x == True:
            if max_circle == True:    
                draw_circle_max = plt.Circle((Cx[i],Cy[i]),maxdist,fill=False)
                axes.add_artist(draw_circle_max)
            if min_circle == True:
                draw_circle_min = plt.Circle((Cx[i],Cy[i]),mindist,fill=False)
                axes.add_artist(draw_circle_min)
    if links == True:  
        for i in range(len(S)):
            for j in range(len(S)):
                if z[i][j].x > 0:
                    plt.plot([Cx[S[i]],Cx[S[j]]],[Cy[S[i]],Cy[S[j]]], color='red')
    plt.show()

def store_indices(x):
    indices = []
    for i in range(len(Cx)):
        if x[i].x == True:
            indices.append(i)
    print('Shops:', end=' ')
    for i in range(len(indices)):
        print(indices[i], end=' ')
    print()

def print_paths(t, z):
    paths=[]
    visto = []
    for l in range(num_tot_camion):
        if t[l].x > 0.99:
            path = []
            counter = 0
            if counter == 0:
                for j in range(len(S)):
                    if z[0][j].x > 0.99:
                        if j not in visto:
                            visto.append(j)
                            path.append(j)
                            counter+=1
                            break
            while(path[-1]!=0):
                for i in range(len(S)):
                    if z[path[-1]][i].x > 0.99:
                        path.append(i)
                        break
            paths.append(path)
    for elem in paths:
        print(f'Path {paths.index(elem)+1}: 0', end=' ')
        for index in elem:
            print(S[index], end=' ')
        print()

def find_path_to_0(index, z):
            path = []
            for j in range(len(S)):
                if z[index][j].x > 0.99:
                    path.append(j)
                    break
            while(True):
                for i in range(len(S)):
                    if z[path[-1]][i].x > 0.99:
                        path.append(i)
                        break
                if i==0:
                    return [True, 0]  
                if i in path[:-1]:
                    return [False, path[0]]
                              



# _________________________________________________
# model

# variables
# xi = 1,0 se costruisco mini-market in i,else

# parameters
# hij=1,0 se casa i lontana massimo maxdist da j , else  for each i,j
# kij=1,0 se casa i è lontana minimo mindist da j, else  for each i,j

# objective function
# min xi*DCi

# constraints
# xi <= usable[i]           for each i
# sumj(xj*hij) >= 1         for each i
# xi+xj <= 2*kij + 1      for each i for each j : i!=j
# x0 = 1

h = []
for i in range(len(Cx)):
    for j in range(len(Cx)):
        if math.sqrt((Cx[i]-Cx[j])**2+(Cy[i]-Cy[j])**2) <= maxdist:
            h.append(1)
        else:
            h.append(0)

k = []
for i in range(len(Cx)):
    for j in range(len(Cx)):
        if math.sqrt((Cx[i]-Cx[j])**2+(Cy[i]-Cy[j])**2) >= mindist:
            k.append(1)
        else:
            k.append(0)

m = mip.Model()
x = np.array([m.add_var(var_type=mip.BINARY) for i in range(len(Cx))])

for i in range(len(Cx)):
    m.add_constr(x[0] >= 1)
    m.add_constr(x[i] <= usable[i])
    m.add_constr(mip.xsum(x[j]*h[i*len(Cx)+j] for j in range(len(Cx))) >= 1)
    for j in range(len(Cx)):
        if i!=j:
            m.add_constr((x[i]+x[j]) <= 2*k[i*len(Cx)+j]+1)
        
m.objective = mip.minimize(mip.xsum(x[i]*Dc[i] for i in range(len(Cx))))

m.optimize()

# problem_plot(Cx, Cy, usable, x, True, min_circle=False)

nmarkets = 0
for i in range(len(Cx)):
    nmarkets += x[i].x


# print(f"RESULT: {m.objective_value} {nmarkets}")

# _________________________________________________
# model routing dei camion

# sets
# i c {i: x[i].x == 1} = S
# 

# variables  
# tl = 1,0 if camion l viene affittato
# yil = 1, 0 if i viene visitato da camion l, altrimenti
# zij = 1, 0 if una camion visita i -> j 
# w >= 0 distanza percorsa da camion l
# fij >= 0 flusso mandato su i->j

# parameters

# objective function
# min suml tl*Fc + Vc*w

# constraints
# sumi(:i!=0) yil <= M*tl               for each l       M = maxstores
# suml yil = 1                          for each i!=0
# suml(y0l) = 1*suml(tl)                               if i=0, for each l 
# sumj z0j = 1*suml(tl)                         for each l              inutile
# sumi zi0 = 1*suml(tl)                          for each l             inutile
# 2*zij <= suml yil + yjl                   for each i,j: i=!j
# sumi sumj(:j!=0) zij = len(S)-1                     for each l                         se ho che camion fa 0->3, riga 1 tutti 0 (somma su tutta la matrice tranne j = 0  = maxstores)
# sumi zij <= suml(yjl)                            for each j,l
# sumj zij <= suml(yil)                             for each i,l
# zij + zji <= 1                               for each i,j i!=0
# suma zja = zij for each i,j aggiungere vincolo che se arrivo in j poi devo partire da j  
# w = sumi sumj math.sqrt((Cx[i]-Cx[j])**2+(Cy[i]-Cy[j])**2)*zij 

# vincoli di flusso per forzare a fare 1 ciclo
# fij <= maxstores*(zij)           for each i,j
# fi(len(S)) <= suml yil              for each i!=0,l
# f0(len(S)) <= 0               
# sumj f0j = len(S)-1               maxstores + (suma(:a!=0) yil-max_stores)            for each l
# sumi fi0 = 0                     
# sumi fi(len(S)) = len(S)-1           maxstores + (suma(:a!=0) yil-max_stores)   for each l
# conservarzione di flusso
# sumi fij - sumt(d in range(len(S)+1)) fjd = 0      for each j!=0 e t


S = np.array([i for i in range(len(Cx)) if x[i].x == 1])
print(len(S))
num_tot_camion = int(np.ceil((len(S)-1)/maxstores))

m2 = mip.Model()

t = np.array([m2.add_var(var_type=mip.BINARY) for l in range(num_tot_camion)])
y = np.array([[m2.add_var(var_type=mip.BINARY) for l in range(num_tot_camion)] for i in range(len(S))])
z = np.array([[m2.add_var(var_type=mip.BINARY)for j in range(len(S))] for i in range(len(S))])
# z = [[[m2.add_var(lb=0,ub=1) for l in range(num_tot_camion)] for j in range(len(S))] for i in range(len(S))]
w = m2.add_var()

f = np.array([[m2.add_var() for j in range(len(S)+1)] for i in range(len(S))])

m2.add_constr(mip.xsum(z[0][j] for j in range(len(S))) == 1*mip.xsum(t[l] for l in range(num_tot_camion)))
m2.add_constr(mip.xsum(np.sqrt(np.power((Cx[S[i]]-Cx[S[j]]), 2)+np.power((Cy[S[i]]-Cy[S[j]]), 2))*z[i][j] for i in range(len(S)) for j in range(len(S))) == w)
m2.add_constr(mip.xsum(z[i][0] for i in range(len(S))) == 1*mip.xsum(t[l] for l in range(num_tot_camion)))
for l in range(num_tot_camion):
    m2.add_constr(mip.xsum(y[i][l] for i in range(len(S)) if i!=0) <= (maxstores)*t[l])

for i in range(len(S)):
    if i!=0:
        m2.add_constr(mip.xsum(y[i][l] for l in range(num_tot_camion)) == 1)
    else:
        for l in range(num_tot_camion):
            m2.add_constr(mip.xsum(y[i][l] for l in range(num_tot_camion)) == mip.xsum(1*t[l] for l in range(num_tot_camion)))
    for j in range(len(S)):
        if i!=j:
            m2.add_constr(2*z[i][j] <= mip.xsum(y[i][l]+y[j][l] for l in range(num_tot_camion)))
        if j != 0 and i != 0:
            m2.add_constr(z[i][j] + z[j][i] <= 1)
            

m2.add_constr(mip.xsum(z[i][j] for j in range(len(S)) for i in range(len(S)) if j!=0) == len(S)-1)
for j in range(len(S)):
    m2.add_constr(mip.xsum(z[i][j] for i in range(len(S))) <= mip.xsum(y[j][l] for l in range(num_tot_camion)))
    m2.add_constr(mip.xsum(z[j][i] for i in range(len(S))) <= mip.xsum(y[j][l] for l in range(num_tot_camion)))

# ### questo vincolo non serve probablmente perchè il flusso lo impone
# for i in range(len(S)):
#     for j in range(len(S)):
#         c1 = [m2.add_constr(mip.xsum(z[j][a] for a in range(len(S)) ) <= z[i][j]+ 20*(1-z[i][j]))]
#         c2 = [m2.add_constr(mip.xsum(z[j][a] for a in range(len(S)) ) >= z[i][j] - 20*(1-z[i][j]))]               

m2.objective = mip.minimize(w*Vc+mip.xsum(t[l]*Fc for l in range(num_tot_camion)))   
m2.optimize()
m2.verbose=0

# problem_plot(Cx, Cy, usable, x, links=True)
continua = True
constraints = []

while True:
    counter=len(S)
    for i in range(len(S)):
        if i !=0:
            out = find_path_to_0(i, z)
            if out[0] == False:
                c = [m2.add_constr(z[i][out[1]]<=0)]
                constraints.append(c)
                # c =  [m2.add_constr(z[out[1]][i]<=0)]
                # constraints.append(c)
                m2.optimize()
                print(m2.objective_value)
                # problem_plot(Cx, Cy, usable, x, links=True)
                break
            else:
                counter -= 1
        # if m2.objective_value==previous_value:
        #     counter-=1
        # if counter == 0:
        #     break
    if counter == 1:
        break

problem_plot(Cx, Cy, usable, x, links=True)

for constraint in constraints:
    m2.remove(constraint)
# m2.remove(c1)
# m2.remove(c2)


######## idea: usare conservazione flusso
m2.add_constr(f[0][len(S)] <= 0)
m2.add_constr(mip.xsum(f[0][j] for j in range(len(S)+1)) == len(S)-1)
m2.add_constr(mip.xsum(f[i][0] for i in range(len(S))) == 0)
m2.add_constr(mip.xsum(f[i][len(S)] for i in range(len(S))) == len(S)-1)
for i in range(len(S)):
    if i != 0:
        m2.add_constr(f[i][len(S)] <= mip.xsum(y[i][l] for l in range(num_tot_camion)))
for i in range(len(S)):
    for j in range(len(S)):  
            m2.add_constr(f[i][j] <= maxstores*z[i][j])
for j in range(len(S)):
    if j != 0:
        m2.add_constr(mip.xsum(f[i][j] for i in range(len(S))) - mip.xsum(f[j][d] for d in range(len(S)+1)) == 0)

for i in range(len(S)):
    for j in range(len(S)):
            m2.start = [(z[i][j],z[i][j].x)]
for l in range(num_tot_camion):
    for i in range(len(S)):
        m2.start = [(t[l],t[l].x),(y[i][l],y[i][l].x),(w,w.x)]

m2.verbose = 1
m2.optimize()


# while True:

#     m2.optimize()
#     print(f"new objective: {m2.objective_value}")

#     pair = None  # Initialize pair to None, leave the loops
#     somma=0
#     for l in range(num_tot_camion):
#         for i in range(len(S)):
#             for j in range(len(S)):
#                 if i !=0 and j!= 0 and i!=j:
#                     if z[i][j].x <= (y[i][l].x+y[j][l].x)/2+eps:
#                         pair = (i,j,l)
#                         break  # Leave the inner loop
#         if pair != None:
#             break  # Leave the outer loop if an inequality is found

#     if pair == None:
#         break  # No violated inequality was found, leave the loop

#     i,j,l = pair
# for j in range(len(S)):
# m2.add_constr()


# m2.optimize()

store_indices(x)
print_paths(t,z)
problem_plot(Cx, Cy, usable, x, links=True)

# for l in range(num_tot_camion):
#     print(f'camion {l} attivo: {t[l].x}', end=' ')
#     print(f'{w.x} visita nodi',end=' ')
#     for i in range(len(S)):
#         if y[i][l].x>=1-0.002:
#             print(f'{S[i]}',end=' ')
#     print()

## print zij e fij
# for i in range(len(S)):
#     for j in range(len(S)):
#         print(z[i][j].x, end=' ')
#     print()
# print()
      
# for i in range(len(S)):
#     for j in range(len(S)+1):
#         print(round(f[i][j].x), end=' ')
#     print()
# print()


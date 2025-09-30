import numpy as np

n=eval(input("请输入节点数量:"))
A=np.identity(n)
A[0][0]=n-1
A[n-1][0]=A[0][n-1]=-1
coeff_res=[float('inf')]*n
coeff_res[0]=1
mat_res=[[[np.zeros((n, n))]] for _ in range(n)]
uniform=[False]

zero_pos=[]
for i in range(1,n-1):
    A[i][0]=A[0][i]=-1
    for j in range(i+1,n):
        if A[i][j]==0:
            zero_pos.append([i,j])
zero_cnt=len(zero_pos)

new_edge=eval(input("请输入图的新增边数"))
coeff_res[1]=2*(n+new_edge-1)

def cha_poly(A):
    eigenvalues, _ = np.linalg.eig(A)
    coefficients=[0]*(n+1)
    coefficients[0]=1
    for num,val in enumerate(eigenvalues,start=1):
        for i in range(num,0,-1):
            coefficients[i]+=val*coefficients[i-1]
    return [np.round(val).real for val in coefficients]

def insert_edge(ed_num,start=0):
    if ed_num==0:
        coefficients=cha_poly(A)
        key1,key2=False,False
        for idx in range(2,n):
            if(coefficients[idx]<coeff_res[idx]):
                key1=True
                coeff_res[idx]=coefficients[idx]
                mat_res[idx]=[A.copy()]
            elif coefficients[idx]==coeff_res[idx]:
                mat_res[idx].append(A.copy())
            else :
                key2=True
        if not key2:
            uniform[0]=True
        elif key1:
            uniform[0]=False
        return
    for i in range(start,zero_cnt):
        if ed_num+i>zero_cnt:
            break
        x,y=zero_pos[i]
        A[x][y]=A[y][x]=-1
        A[x][x]+=1
        A[y][y]+=1
        if A[x][x]<=A[x-1][x-1] and A[y][y]<=A[y-1][y-1]:
            insert_edge(ed_num-1,i+1)
        A[x][y]=A[y][x]=0
        A[x][x]-=1
        A[y][y]-=1
        
insert_edge(new_edge)

print("uniform[0]={}".format(uniform[0]))
print("ci={}".format(coeff_res))

show=input("显示满足ci最小下的所有矩阵输入all,显示其中之一输入任意字符,显示对应最小拉普拉斯系数的矩阵输入whole,不显示请直接回车")
if show!="":
    if show=="all":
        print(coeff_res[(n+1)//2])
        for k in mat_res[(n+1)//2]:
            print(k)
    elif show=="whole":
        for i in range(2,n):
            print("c{}={}".format(i,coeff_res[i]))
            for j in mat_res[i]:
                print(j)
    else:
        print(coeff_res[(n+1)//2])
        print(mat_res[(n+1)//2][0])
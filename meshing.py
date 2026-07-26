import cv2
import numpy as np
import triangle
from scipy.sparse import  coo_matrix,diags,csr_matrix
import scipy.sparse.linalg as spla

#  creating a mesh


image=cv2.imread('test3.png')
image=cv2.resize((image),(800,600))
image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
_,image=cv2.threshold(image,127,255,cv2.THRESH_BINARY)
contours,_=cv2.findContours(image,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
contour=max(contours,key=cv2.contourArea)
# print(f"total vertices are {len(contour)}")
image=cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
image2=image.copy()
cv2.drawContours(image,[contour],-1,(0,255,0),3)
cv2.imshow("contour",image)
i=1
approx=cv2.approxPolyDP(contour,i,True)
# print(f"\n,total vertices are{len(approx)}")
cv2.polylines(image2,[approx],True,(0,255,0),3)
cv2.imshow("approx",image2)
image3=image2.copy()
image4=image2.copy()
# cv2.drawContours(image2,[approx],-1,(255,0,0),3)
# cv2.imshow("reduced-boundary",image2)
# cv2.waitKey(20000)
approx= approx.reshape(-1, 2)
vertices=len(approx)
edges=[]
for i in range(vertices):
    j=(i+1)%vertices
    edges.append([i,j])
edge=np.array(edges)
approx=approx.astype(float)
graph={'vertices':approx,'segments':edge}
# print(approx)
result=triangle.triangulate(graph, 'pq15a100')
# print(result.keys())
for x in result['triangles']:
    p=result['vertices'][x].astype(np.int32)
    cv2.polylines(image2,[p],True,(255,0,0),1)
cv2.imshow("mesh",image2)
cv2.waitKey(20000)

# print(result['triangles'].shape)
# print(result['triangles'][0])
# print(result['vertices'].shape)    
# print("\m new")




# contraction


def build_L(V, F):      # uniform Laplacian ( provided better results then the contangent laplacian one)
    N = len(V)
    i, j, k = F[:, 0], F[:, 1], F[:, 2]

    rows = np.concatenate([i, j, j, k, k, i])
    cols = np.concatenate([j, i, k, j, i, k])
    data = np.ones(len(rows), dtype=float)

    L_off = csr_matrix((data, (rows, cols)), shape=(N, N))
    L_off.data[:] = 1.0  # Adjacency

    row_sums = np.array(L_off.sum(axis=1)).flatten()
    L_diag = csr_matrix((-row_sums, (np.arange(N), np.arange(N))), shape=(N, N))
    return L_off + L_diag

# def build_L(V, F):                   // Contangent laplacian method based on geometry and understand the mesh better than the uniform one
#     N = len(V)
#     i, j, k = F[:, 0], F[:, 1], F[:, 2]
#     vi, vj, vk = V[i], V[j], V[k]
#     # Cotangent = cos / sin (dot product / 2D cross product)
#     def cot_vec(a, b, c):
#         u, v = a - c, b - c
#         cosang = np.sum(u * v, axis=1)
#         sinang = np.abs(u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0])
#         sinang = np.maximum(sinang, 1e-10) # Avoid divide-by-zero
#         return np.clip(cosang / sinang ,-10.0,10.0)
#     # Half-cotangents opposite to edges
#     wk = cot_vec(vi, vj, vk) / 2.0  # opposite edge (i, j)
#     wi = cot_vec(vj, vk, vi) / 2.0  # opposite edge (j, k)
#     wj = cot_vec(vk, vi, vj) / 2.0  # opposite edge (k, i)
#     rows = np.concatenate([i, j, j, k, k, i])
#     cols = np.concatenate([j, i, k, j, i, k])
#     data = np.concatenate([wk, wk, wi, wi, wj, wj])
#     L_off = csr_matrix((data, (rows, cols)), shape=(N, N))
#     row_sums = np.array(L_off.sum(axis=1)).flatten()
#     L_diag = csr_matrix((-row_sums, (np.arange(N), np.arange(N))), shape=(N, N))
#     L = L_off + L_diag
#     return L

def vertex_areas(V, F):
    N = len(V)
    i, j, k = F[:,0], F[:,1], F[:,2]
    p0, p1, p2 = V[i], V[j], V[k]
    area = 0.5*np.abs(p0[:,0]*(p1[:,1]-p2[:,1]) + p1[:,0]*(p2[:,1]-p0[:,1]) + p2[:,0]*(p0[:,1]-p1[:,1]))
    area_safe=np.maximum(np.abs(area),1e-6)
    A = np.zeros(N)
    np.add.at(A, i, area_safe/3)
    np.add.at(A, j, area_safe/3)
    np.add.at(A, k, area_safe/3)
    return A

#CALCULATE THE TOTAL AREA
      
def face_areas(V, F):
    i, j, k = F[:,0], F[:,1], F[:,2]
    p0, p1, p2 = V[i], V[j], V[k]
    return 0.5*np.abs(p0[:,0]*(p1[:,1]-p2[:,1]) + p1[:,0]*(p2[:,1]-p0[:,1]) + p2[:,0]*(p0[:,1]-p1[:,1]))


# IT CONTRACTS THE VERTICES BY USING LAPLACING MATRIX ALONG WITH THE WEIGHTS 

def contract(Wl,Wh,V_next,L,V_cur,Gamma=0.5):
        V_cur=(0.8*V_cur)+(0.20*V_next)
        W_L2 = diags(Wl**2)
        W_H2 = diags(Wh**2)
        A = L.T @ W_L2 @ L + W_H2
        # print("\n",A)
        # B = W_H2 @ V_next
        B = W_H2 @ V_cur
        # print(B.shape) 
        V_next = spla.spsolve(A.tocsr(), B)
        # V_updated = (1.0 - Gamma) * V_cur + Gamma * V_next
        # print(V_next,"\n",V_next.shape)
        return V_next

# def update_deformation_weights(Wh,V1,F,Wh0,Omega_0, Omega_Ct, Wl_prev, s_L=1.0, Wl_cap=50, Wh_cap=1000.0,alpha=0.4,Wh_floor=0.05):
#      Wh_scaled = Wh0 * np.sqrt(Omega_0 / np.maximum(np.abs(Omega_Ct), 1e-6))
#      Wh_target = np.minimum(Wh_scaled, Wh_cap)
#      Wh_curr = (alpha * Wh_target) + ((1.0 - alpha) * Wh)

#     #  Wl_scaled = Wl_prev * np.sqrt(np.mean(face_areas(V1,F)))
#     #  Wl_curr = np.minimum(Wl_scaled, Wl_cap)
#      Wl_scaled = Wl_prev *s_L
#      Wl_curr = (Wl_scaled)
#      Wh = np.minimum(Wh_target, Wh * 1.3)
#      return Wl_curr, Wh_curr



# UPDATE WEIGHTS IS VERY IMPORTANT FOR THIS ALFORITHM , THE CRUX IS TO FIND A BALANCE BETWEEN BOTH THE FORCES AND SHRINK THE MESH INTO THE SMOOTH 1D SKELETON

def update_weights(Wl, Wh, Wh0, Omega_0, Omega_t, s_L,Wl_cap=100, Wh_cap=50.0, max_growth=1.3):
    Wl_next = Wl * s_L
    # Wl_next=np.minimum(Wl_next,Wl_cap)

    Wh_next = Wh0 * np.sqrt(Omega_0 / np.maximum(Omega_t, 1e-8))

    # Wh_target = Wh0 * np.sqrt(Omega_0 / np.maximum(Omega_t, 1e-8))
    # Wh_target = np.minimum(Wh_target, Wh_cap)
    # Wh_next = np.minimum(Wh_target, Wh * max_growth)
    # Wh_next=np.clip(Wh_next,0.0005,Wh_cap)   # can grow at most 30% per step
    return Wl_next, Wh_next




 # YOU CAN RUN THE CODE WITHOUT USING THIS TOO , IT IS JUST USED TO A SMOOTH RESULT , BUT CURRENTLY THIS FUCNTION DOESNT MAKE MUCH DIFFERNCE SO IF YOU WANT YOU CAN REMOVE IT

def smooth_omega(Omega, F, iterations=2):          
    N = len(Omega)
    edges = np.vstack([F[:,[0,1]], F[:,[1,2]], F[:,[2,0]]])
    for _ in range(iterations):
        s = np.zeros(N); c = np.zeros(N)
        np.add.at(s, edges[:,0], Omega[edges[:,1]]); np.add.at(c, edges[:,0], 1)
        np.add.at(s, edges[:,1], Omega[edges[:,0]]); np.add.at(c, edges[:,1], 1)
        Omega = (Omega + s) / (1 + c)
    return Omega




  # NOT A PART OF THE CONTRACTION MESH ALGORITHM I JUST ADDED TO FIND THE EDGES WHERE I WAS HAVIGN PROBLME YOU MAY NOT NEED TO USE THIS AND CAN REMOVE IT DEPENDING UPON YOU


def find_collapsed_edges(V, F, V_original, shrink_threshold=0.05):   
    """
    Detect edges that have shrunk to less than `shrink_threshold` of their
    ORIGINAL length — this measures actual collapse, not just small size.
    """
    edges = np.vstack([F[:, [0,1]], F[:, [1,2]], F[:, [2,0]]])
    edges = np.unique(np.sort(edges, axis=1), axis=0)

    orig_lens = np.linalg.norm(V_original[edges[:,0]] - V_original[edges[:,1]], axis=1)
    curr_lens = np.linalg.norm(V[edges[:,0]] - V[edges[:,1]], axis=1)

    orig_lens_safe = np.maximum(orig_lens, 1e-8)
    shrink_ratio = curr_lens / orig_lens_safe

    collapsed_mask = shrink_ratio < shrink_threshold
    collapsed_edges = edges[collapsed_mask]

    print(f"total edges: {len(edges)}")
    print(f"collapsed (shrunk to <{shrink_threshold*100:.0f}% of original): {len(collapsed_edges)}  ({100*len(collapsed_edges)/len(edges):.1f}%)")
    print(f"shrink ratio range: min={shrink_ratio.min():.4f} max={shrink_ratio.max():.4f} mean={shrink_ratio.mean():.4f}")

    return collapsed_edges, shrink_ratio, edges


 # NOT A PART OF THE CONTRACTION MESH ALGORITHM, I  ADDED THIS FUNTION TO FIND THE EDGES WHERE I WAS HAVIGN PROBLEM .YOU MAY NOT NEED TO USE THIS AND CAN REMOVE IT DEPENDING UPON YOU


def find_exploding_edges(V, F, V_original, grow_threshold=3.0):      
    edges = np.vstack([F[:, [0,1]], F[:, [1,2]], F[:, [2,0]]])
    edges = np.unique(np.sort(edges, axis=1), axis=0)

    orig_lens = np.linalg.norm(V_original[edges[:,0]] - V_original[edges[:,1]], axis=1)
    curr_lens = np.linalg.norm(V[edges[:,0]] - V[edges[:,1]], axis=1)
    ratio = curr_lens / np.maximum(orig_lens, 1e-8)

    exploding_mask = ratio > grow_threshold
    exploding_edges = edges[exploding_mask]
    exploding_verts = np.unique(exploding_edges)

    print(f"exploding edges (>{grow_threshold}x original): {len(exploding_edges)}")
    if len(exploding_verts) > 0:
        print(f"vertex positions involved (current, pixel space):")
        print(V[exploding_verts][:10])   # first 10, to avoid flooding output

    return exploding_edges, exploding_verts



# MAIN CODE STARTS FROM HERE 


V=np.array(result["vertices"],dtype=float)
V1=V.copy()
F=np.array(result["triangles"],dtype=int)
N=len(V)
V2=V.copy()
total_area_t = face_areas(V, F).sum()
Wl = np.full(N, 1e-3*(np.sqrt(np.mean(total_area_t))))
Wh = np.full(N, 1.0)
Wh0=Wh.copy()
Wl_matrix=diags(Wl)
Wh_matrix=diags(Wh)
Omega_0_raw = vertex_areas(V, F)
Omega_0 = smooth_omega(Omega_0_raw, F, iterations=2)
Omega_0 = np.maximum(Omega_0, 0.1 * Omega_0.mean())

  #global mesh area
print("checking original mesh\n====")
find_collapsed_edges(V1, F,V)


image_frames={}
i=6
for step in range(20):
     image_frames[i]=image4.copy()
     L=build_L(V1,F)
     V1=contract(Wl,Wh,V1,L,V2)
     Omega_t = smooth_omega(vertex_areas(V1, F), F, iterations=2)
     Wl,Wh=update_weights(Wl,Wh,Wh0,Omega_0,Omega_t,s_L=1.5,Wh_cap=50.0,)
     disp = np.linalg.norm(V1 - V, axis=1)
     area_ratio = face_areas(V1, F).sum() / total_area_t
     V2=V1
     
    #  print(f"step={step:2d}  area_ratio={area_ratio:.4e}  "f"Wl=[{Wl.min():.3g},{Wl.max():.3g}]  "     f"Wh=[{Wh.min():.3g},{Wh.max():.3g}]  "f"max_disp={disp.max():.2f}  mean_disp={disp.mean():.2f}")
    #  print(f"\n{step} iteration ")
    #  find_collapsed_edges(V1, F,V)
    #  print(f"/n exploding edges iteration{step}")
    #  find_exploding_edges(V1, F, V, grow_threshold=3.0)
     
     edges = np.vstack([F[:, [0,1]], F[:, [1,2]], F[:, [2,0]]])
     edges = np.unique(np.sort(edges, axis=1), axis=0)
     orig_lens = np.linalg.norm(V[edges[:,0]] - V[edges[:,1]], axis=1)
     curr_lens = np.linalg.norm(V1[edges[:,0]] - V1[edges[:,1]], axis=1)
     ratio = curr_lens / np.maximum(orig_lens, 1e-8)
     n_exploding = np.sum(ratio > 3.0)

     area_ratio = face_areas(V1, F).sum() / total_area_t
     print(f"step={step:2d} area_ratio={area_ratio:.4e} exploding_edges={n_exploding} "
          f"Wl_max={Wl.max():.3g} Wh_max={Wh.max():.3g}")
     
     for x in result['triangles']:
                   p=V1[x].astype(np.int32)
                   cv2.polylines(image_frames[i],[p],True,(255,0,0),1)
     cv2.imshow(f"mesh{i}",image_frames[i])
    #  cv2.imwrite(f'mesh{step}.png',image_frames[i])
     cv2.waitKey(20000)
     i=i+1



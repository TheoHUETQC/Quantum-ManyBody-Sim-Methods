using ITensors

function tensor_identity(; d=4, i=i, j=j)
    Id = ITensor(i,j)
    for n=1:d
        Id[i=>n,j=>n] = 1
    end
    Id
end

d = 2 

i = Index(d,"index_i")
j = Index(d,"index_j")
Id = tensor_identity(d=d, i=i, j=j)
@show Id
U,S,V = svd(Id,(i,j))
A, B = U*S, V

println("A inds = ",inds(A))
println("B inds = ",inds(B))

k = Index(d,"index_k")
l = Index(d,"index_l")
Id2 = tensor_identity(d=d, i=k, j=l)
Up, Sp, Vp = svd(Id2,(k,l))
C, D = Up, Sp*Vp

println("C inds = ",inds(C))
println("D inds = ",inds(D))

println("U :")
@show U

println("S :")
@show S

println("V :")
@show V
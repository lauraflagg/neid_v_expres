### A Pluto.jl notebook ###
# v0.20.18

using Markdown
using InteractiveUtils

# ╔═╡ 0dd73708-72bd-4849-a774-e76007fd82b5
using HDF5

# ╔═╡ e51dee64-9b37-49ad-ad4b-8a1b25d9f0e7
using Printf

# ╔═╡ 97f51b55-83c9-439b-8b78-a83f1d09718d


# ╔═╡ 143b0881-b4b6-48b1-a308-d667818dbfeb
"""
Read a list-of-arrays stored under group `g[name]` where datasets are named 0000, 0001, ...
Returns Vector{Any} (each element can be 1D or 2D array depending on what was written).
"""
function read_list_of_arrays(g::HDF5.Group, name::AbstractString)
    sub = g[name]
    n = Int(read(attributes(sub)["length"]))
    out = Vector{Any}(undef, n)
    for i in 0:n-1
        key = @sprintf("%04d", i)
        out[i+1] = read(sub[key])
    end
    return out
end

# ╔═╡ df54d8e5-86c8-4fb2-b726-d1c488038517
"""
Read date_sets stored under group `g["date_sets"]` with subgroups 0000,0001,... each containing:
  - jds (Float64 vector)
  - file_locs (String vector)
Returns Vector{NamedTuple} with fields (:jds, :file_locs).
"""
function read_date_sets(g::HDF5.Group)
    ds = g["date_sets"]
    n = Int(read(attributes(ds)["length"]))
    out = Vector{NamedTuple{(:jds, :file_locs), Tuple{Vector{Float64}, Vector{String}}}}(undef, n)

    for i in 0:n-1
        key = @sprintf("%04d", i)
        w = ds[key]
        jds = Vector{Float64}(read(w["jds"]))
        file_locs = Vector{String}(read(w["file_locs"]))  # vlen strings become Vector{String}
        out[i+1] = (jds=jds, file_locs=file_locs)
    end
    return out
end

# ╔═╡ e2eaa791-d79b-40fa-996e-dc69f2e7f90e
"""
Read one instrument block (NEID or EXPRES).
Returns a NamedTuple with wls, coadds, uncs, date_sets.
"""
function read_instrument(f::HDF5.File, inst::AbstractString)
    g = f[inst]
    wls      = read_list_of_arrays(g, "wls")
    coadds   = read_list_of_arrays(g, "coadds")
    uncs     = read_list_of_arrays(g, "uncs")
    date_sets = read_date_sets(g)
    return (wls=wls, coadds=coadds, uncs=uncs, date_sets=date_sets)
end

# ╔═╡ a8e6dc1d-7338-4875-9e69-71b75c7f886a
"""
Safely collect file-level attributes into a Dict{String,Any}.
HDF5.jl attributes are not directly iterable, so we iterate keys(attributes(obj)).
"""
function read_attrs(obj)::Dict{String,Any}
    a = attributes(obj)
    d = Dict{String,Any}()
    for k in keys(a)
        # read(a[k]) usually gives String / numbers; sometimes a 0-d array or bytes
        v = read(a[k])
        d[string(k)] = v
    end
    return d
end

# ╔═╡ d2b9dd99-1c2a-405b-9bdb-31efbe5a86a4
"""
Top-level reader for the whole file.
Returns a NamedTuple: (overlaps, NEID, EXPRES, attrs)
"""
function read_coadds_hdf5(path::AbstractString)
    h5open(path, "r") do f
        overlaps = haskey(f, "overlaps") ? read(f["overlaps"]) : nothing
        neid = read_instrument(f, "NEID")
        expres = read_instrument(f, "EXPRES")
        attrs = read_attrs(f)
        return (overlaps=overlaps, NEID=neid, EXPRES=expres, attrs=attrs)
    end
end

# ╔═╡ 7d3bed5a-79ff-4431-9bfe-9abd519bd1ce
data=read_coadds_hdf5("coadds_outputs.h5")

# ╔═╡ a0d5900c-8945-4eb9-9317-83881a9c435f


# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
HDF5 = "f67ccb44-e63f-5c2f-98bd-6dc0ccc4ba2f"
Printf = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[compat]
HDF5 = "~0.17.2"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.11.6"
manifest_format = "2.0"
project_hash = "f62dc1d910b29daf0321112cebb6193c63d51430"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.2"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
version = "1.11.0"

[[deps.Compat]]
deps = ["TOML", "UUIDs"]
git-tree-sha1 = "0037835448781bb46feb39866934e243886d756a"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.18.0"

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

    [deps.Compat.weakdeps]
    Dates = "ade2ca70-3891-5945-98fb-dc099432e06a"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.1.1+0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"
version = "1.11.0"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"
version = "1.11.0"

[[deps.HDF5]]
deps = ["Compat", "HDF5_jll", "Libdl", "MPIPreferences", "Mmap", "Preferences", "Printf", "Random", "Requires", "UUIDs"]
git-tree-sha1 = "e856eef26cf5bf2b0f95f8f4fc37553c72c8641c"
uuid = "f67ccb44-e63f-5c2f-98bd-6dc0ccc4ba2f"
version = "0.17.2"

    [deps.HDF5.extensions]
    MPIExt = "MPI"

    [deps.HDF5.weakdeps]
    MPI = "da04e1cc-30fd-572f-bb4f-1f8673147195"

[[deps.HDF5_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LazyArtifacts", "LibCURL_jll", "Libdl", "MPICH_jll", "MPIPreferences", "MPItrampoline_jll", "MicrosoftMPI_jll", "OpenMPI_jll", "OpenSSL_jll", "TOML", "Zlib_jll", "libaec_jll"]
git-tree-sha1 = "e94f84da9af7ce9c6be049e9067e511e17ff89ec"
uuid = "0234f1f7-429e-5d53-9886-15a909be8d59"
version = "1.14.6+0"

[[deps.Hwloc_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "92f65c4d78ce8cdbb6b68daf88889950b0a99d11"
uuid = "e33a78d0-f292-5ffc-b300-72abe9b543c8"
version = "2.12.1+0"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "0533e564aae234aff59ab625543145446d8b6ec2"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.7.1"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"
version = "1.11.0"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.6.0+0"

[[deps.LibGit2]]
deps = ["Base64", "LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"
version = "1.11.0"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.7.2+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.0+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"
version = "1.11.0"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"
version = "1.11.0"

[[deps.MPICH_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Hwloc_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "MPIPreferences", "TOML"]
git-tree-sha1 = "d72d0ecc3f76998aac04e446547259b9ae4c265f"
uuid = "7cb0a576-ebde-5e09-9194-50597f1243b4"
version = "4.3.1+0"

[[deps.MPIPreferences]]
deps = ["Libdl", "Preferences"]
git-tree-sha1 = "c105fe467859e7f6e9a852cb15cb4301126fac07"
uuid = "3da0fdf6-3ccc-4f1b-acd9-58baa6c99267"
version = "0.1.11"

[[deps.MPItrampoline_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "MPIPreferences", "TOML"]
git-tree-sha1 = "e214f2a20bdd64c04cd3e4ff62d3c9be7e969a59"
uuid = "f1f71cc9-e9ae-5b93-9b94-4fe0e1ad3748"
version = "5.5.4+0"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"
version = "1.11.0"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.6+0"

[[deps.MicrosoftMPI_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bc95bf4149bf535c09602e3acdf950d9b4376227"
uuid = "9237b28f-5490-5468-be7b-bb81f5f5e6cf"
version = "10.1.4+3"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"
version = "1.11.0"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2023.12.12"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.OpenMPI_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Hwloc_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "MPIPreferences", "TOML", "Zlib_jll"]
git-tree-sha1 = "ec764453819f802fc1e144bfe750c454181bd66d"
uuid = "fe0851c0-eecd-5654-98d4-656369965a5c"
version = "5.0.8+0"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "2ae7d4ddec2e13ad3bddf5c0796f7547cf682391"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.5.2+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "Random", "SHA", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.11.0"

    [deps.Pkg.extensions]
    REPLExt = "REPL"

    [deps.Pkg.weakdeps]
    REPL = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "0f27480397253da18fe2c12a4ba4eb9eb208bf3d"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.5.0"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"
version = "1.11.0"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
version = "1.11.0"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "62389eeff14780bfe55195b7204c0d8738436d64"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.1"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"
version = "1.11.0"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"
version = "1.11.0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+1"

[[deps.libaec_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "f5733a5a9047722470b95a81e1b172383971105c"
uuid = "477f73a3-ac25-53e9-8cc3-50b2fa2566f0"
version = "1.1.3+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.59.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+2"
"""

# ╔═╡ Cell order:
# ╠═97f51b55-83c9-439b-8b78-a83f1d09718d
# ╠═0dd73708-72bd-4849-a774-e76007fd82b5
# ╠═e51dee64-9b37-49ad-ad4b-8a1b25d9f0e7
# ╠═143b0881-b4b6-48b1-a308-d667818dbfeb
# ╠═df54d8e5-86c8-4fb2-b726-d1c488038517
# ╠═e2eaa791-d79b-40fa-996e-dc69f2e7f90e
# ╠═a8e6dc1d-7338-4875-9e69-71b75c7f886a
# ╠═d2b9dd99-1c2a-405b-9bdb-31efbe5a86a4
# ╠═7d3bed5a-79ff-4431-9bfe-9abd519bd1ce
# ╠═a0d5900c-8945-4eb9-9317-83881a9c435f
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002

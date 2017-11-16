### plot results from SPCA comparison ###
# There are two plots and three algos to compare
# "r" refers to the SPC function from the PMA package (see Witten, Tibshirani, and Hastie 2009)
# "j" refers to the two proximal distance routines for sparse PCA
# the first PD SPCA routine projects the entire matrix of loadings to a single sparse constraint
# the second PD SPCA routine projects each loading to its own sparse constraint 


plot.spca.results = function(){
    # load data
    j       = read.table("spca_results_k25_sumabsv8_julia.txt", header=T)
    a       = read.table("spca_results_k25_noaccel.txt", header=T)
    r       = read.table("spca_results_k25_sumabsv8_R.txt", header=F)
    j1.pve  = j$PVE[1:25]       # PVEs from matrix projection
    j2.pve  = j$PVE[26:50]      # from columnwise projection
    a1.pve  = a$PVE[1:25]       # PVEs from unaccel matrix proj
    a2.pve  = a$PVE[26:50]       # PVEs from unaccel matrix proj
    r.pve   = r$V8
    pcs     = j$PCs[1:25]
    j1.time = j$Time[1:25] + 2  # why +2? full SVD warm-start on data in Julia requires 2 seconds
    j2.time = j$Time[26:50] + 2 # why +2? full SVD warm-start on data in Julia requires 2 seconds
    a1.time = a$Time[1:25] + 2  # why +2? full SVD warm-start on data in Julia requires 2 seconds
    a2.time = a$Time[26:50] + 2 # why +2? full SVD warm-start on data in Julia requires 2 seconds
    r.time  = r$V5

    # plot for compute times needs argument for 'ylim' 
    times   = c(0, max(max(as.numeric(j1.time)), max(as.numeric(r.time))))

    # output file names?
    outfile_pve  = "spca_results_k25_sumabsv8_noaccel_pve.pdf"
    outfile_time = "spca_results_k25_sumabsv8_noaccel_time.pdf"

    # initialize plot for PVE
    pdf(outfile_pve)

    # plot first aPD results (matrix projection) 
    plot(
        y    = j1.pve, 
        x    = pcs, 
        type = "l", 
        lty  = 1,
        lwd  = 3,
        col  = "blue",
        main = "PVE of q Sparse PCs",
#        sub  = "|| v ||_1 = 8", 
        xlab = "q", 
        ylab = "PVE", 
        xlim = c(1,25)
    )

    # overlay second aPD results (columwise projection)
    lines(
        y    = j2.pve,
        x    = pcs,
        type = "l",
        lty  = 2, 
        lwd  = 3,
        col  = "black"
    )

    # overlay first PD results (mat proj)
    lines(
        y    = a1.pve,
        x    = pcs,
        type = "l",
        lty  = 3, 
        lwd  = 3,
        col  = "red"
    )

    # overlay second PD results (mat proj)
    lines(
        y    = a2.pve,
        x    = pcs,
        type = "l",
        lty  = 4, 
        lwd  = 3,
        col  = "darkgreen"
    )

    # overlay PMA results
    lines(
        y    = r.pve,
        x    = pcs,
        type = "l",
        col  = "chocolate4",
        lty  = 5, 
        lwd  = 3 
    )


    # add legend
    legend("topleft", c("aPD1", "aPD2", "PD1", "PD2", "SPC"), lty=c(1,2,3,4,5), lwd=c(3,3,3,3,3), col=c("blue", "black", "red", "darkgreen", "chocolate4"))

    # close plot file
    dev.off()

    # do same for compute times
    pdf(outfile_time)
    plot(
        y    = j1.time, 
        x    = pcs, 
        type = "l", 
        lty  = 1,
        lwd  = 3,
        col  = "blue",
        main = "Compute time to calculate q Sparse PCs", 
#        sub  = "|| v ||_1 = 8", 
        xlab = "q", 
        ylab = "Compute time", 
        xlim = c(1,25),
        ylim = times
    )
    lines(
        y    = j2.time,
        x    = pcs,
        type = "l",
        lty  = 2, 
        lwd  = 3,
        col  = "black"
    )
    lines(
        y    = a1.time,
        x    = pcs,
        type = "l",
        lty  = 3, 
        lwd  = 3,
        col  = "red"
    )

    # overlay second PD results (mat proj)
    lines(
        y    = a2.time,
        x    = pcs,
        type = "l",
        lty  = 4, 
        lwd  = 3,
        col  = "darkgreen"
    )

    lines(
        y    = r.time,
        x    = pcs,
        type = "l",
        col  = "chocolate4",
        lwd  = 3,
        lty  = 5
    )

    # add legend
    legend("topleft", c("aPD1", "aPD2", "PD1", "PD2", "SPC"), lty=c(1,2,3,4,5), lwd=c(3,3,3,3,3), col=c("blue", "black", "red", "darkgreen", "chocolate4"))

    # close plot file
    dev.off()

    return()
}

plot.spca.results()

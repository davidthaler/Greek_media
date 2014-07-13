#NB: call loadDV.R first

y.dv <- function(npos = 100, nneg=1000, t1=-0.3, t2=0.1){
  dv.neg <- dv[y==0 & dv != -99]
  dv.pos <- dv[y==1 & dv != -99]
  z.neg <- z[y==0 & dv != -99]
  z.pos <- z[y==1 & dv != -99]
  row.pos <- row.dv[y==1 & dv != -99]
  row.neg <- row.dv[y==0 & dv != -99]
  pos.idx <- sample(length(dv.pos), size=npos, replace=FALSE)
  neg.idx <- sample(length(dv.neg), size=nneg, replace=FALSE)
  dv.pos <- dv.pos[pos.idx]
  z.pos <- z.pos[pos.idx]
  row.pos <- row.pos[pos.idx]
  dv.neg <- dv.neg[neg.idx]
  z.neg <- z.neg[neg.idx]
  row.neg <- row.neg[neg.idx]
  f1 <- data.frame(dv=dv.pos, ratio=z.pos, y=1, row=(row.pos >= -t2))
  f2 <- data.frame(dv=dv.neg, ratio=z.neg, y=0, row=(row.neg >= -t2))
  f <- rbind(f1,f2)
  plot(f$dv, 
       log(f$ratio), 
       xlab='dv', 
       ylab='log.ratio', 
       col=2*f$y+2, 
       pch=1 + 21*f$row)
  abline(v=t1)
  f
}

by.class <- function(n=1000){
  #TODO get a subsample of neg and pos rows separately
  idx = sample(nrow(dv), n)
  for(k in 1:ncol(y)){
    #TODO: trim out -99's
    plot(dv[idx, k], 
         row.dv[idx,k], 
         xlab='dv', 
         ylab='row.dv', 
         col=2 * y[idx, k] + 2)
    input <- readline("Press Q to quit, any other key to continue.")
    if(input=='Q') break
  }
}
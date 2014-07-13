require(data.table)

#NB: expects headers
dv <- fread('../../artifacts/cvdv.csv', header=TRUE)
y <- fread('../../artifacts/y.csv', header=TRUE)
dv <- as.data.frame(dv)
y <- as.data.frame(y)

class.counts <- colSums(y)
class.ratio <- class.counts / max(class.counts)
rowmax <- apply(dv, 1, max)
colmax <- apply(dv, 2, max)
row.dv <- dv - rowmax
col.dv <- t(t(dv) - colmax)
z <- 0*y
z <- t(t(z) + class.ratio)
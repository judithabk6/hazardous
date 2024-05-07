#conda activate r_env_survival
# cd /data/parietal/store/work/jabecass/survival/hazardous/benchmark/
library("randomForestSRC")


for (seed in 0:4) {
    df = read.table(paste('seer_srf_50000_', format(seed, scientific=F), '.csv', sep=''), header=T, sep=',')
    df[sapply(df, is.character)] <- lapply(df[sapply(df, is.character)], 
                                           as.factor)
    start.time <- Sys.time()
    df.obj <- rfsrc(Surv(duration, event) ~ ., df)
    end.time <- Sys.time()
    time.taken <- round(end.time - start.time,5)
    save(df.obj, file=paste('result_seer_srf_50000_', format(seed, scientific=F), '.rda', sep=''))
    print(time.taken)
    save(time.taken, file=paste('time_seer_srf_50000_', format(seed, scientific=F), '.Rdata', sep=''))

}


for (seed in 0:4) {
    df = read.table(paste('seer_srf_100000_', format(seed, scientific=F), '.csv', sep=''), header=T, sep=',')
    df[sapply(df, is.character)] <- lapply(df[sapply(df, is.character)], 
                                           as.factor)
    start.time <- Sys.time()
    df.obj <- rfsrc(Surv(duration, event) ~ ., df)
    end.time <- Sys.time()
    time.taken <- round(end.time - start.time,5)
    save(df.obj, file=paste('result_seer_srf_100000_', format(seed, scientific=F), '.rda', sep=''))
    print(time.taken)
    save(time.taken, file=paste('time_seer_srf_100000_', format(seed, scientific=F), '.Rdata', sep=''))

}



for (seed in 0:4) {
df = read.table(paste('seer_srf_None_', format(seed, scientific=F), '.csv', sep=''), header=T, sep=',')
df[sapply(df, is.character)] <- lapply(df[sapply(df, is.character)], 
                                       as.factor)
start.time <- Sys.time()
df.obj <- rfsrc(Surv(duration, event) ~ ., df)
end.time <- Sys.time()
time.taken <- round(end.time - start.time,5)
save(df.obj, file=paste('result_seer_srf_None_', format(seed, scientific=F), '.rda', sep=''))
print(time.taken)
save(time.taken, file=paste('time_seer_srf_None_', format(seed, scientific=F), '.Rdata', sep=''))
}


for (seed in 0:4) {
    df_test = read.table(paste('seer_test_', format(seed, scientific=F), '.csv', sep=''), header=T, sep=',')
    load(paste('result_seer_srf_50000_', format(seed, scientific=F), '.rda', sep=''))
    pred_obj = predict(object=df.obj, df_test)
    write.table(pred_obj$cif, paste('test_cif_seer_srf_50000_', format(seed, scientific=F), '.csv', sep=''), row.names=F, sep=',')
    write.table(pred_obj$time.interest, paste('test_timegrid_seer_srf_50000_', format(seed, scientific=F), '.csv', sep=''), row.names=F, sep=',')
}

for (seed in 0:4) {
    df_test = read.table(paste('seer_test_', format(seed, scientific=F), '.csv', sep=''), header=T, sep=',')
    load(paste('result_seer_srf_50000_', format(seed, scientific=F), '.rda', sep=''))
    pred_obj = predict(object=df.obj, df_test)
    write.table(pred_obj$cif, paste('test_cif_seer_srf_50000_', format(seed, scientific=F), '.csv', sep=''), row.names=F, sep=',')
    write.table(pred_obj$time.interest, paste('test_timegrid_seer_srf_50000_', format(seed, scientific=F), '.csv', sep=''), row.names=F, sep=',')
}

for (seed in 0:4) {
    df_test = read.table(paste('seer_test_', format(seed, scientific=F), '.csv', sep=''), header=T, sep=',')
    load(paste('result_seer_srf_100000_', format(seed, scientific=F), '.rda', sep=''))
    pred_obj = predict(object=df.obj, df_test)
    write.table(pred_obj$cif, paste('test_cif_seer_srf_100000_', format(seed, scientific=F), '.csv', sep=''), row.names=F, sep=',')
    write.table(pred_obj$time.interest, paste('test_timegrid_seer_srf_100000_', format(seed, scientific=F), '.csv', sep=''), row.names=F, sep=',')
}

## This script will produce the final output and write it in a csv in current directory.
## Cross Validation Codes have not been included.
## Some Libraries will be required.

## Uncomment the following if installation is required. These packages also install some
## Dependencies
# install.packages(c("caret",
#                    "xgboost",
#                    "gdata",
#                    "randomForest",
#                    "forecast",
#                    "doParallel"))

library(caret)
library(xgboost)
library(gdata)
library(randomForest)
library(forecast)
library(doParallel)


################################################################################################

## Data Preparation
# Reading the files
static_data=read.csv("ML_Bond_metadata.csv")
ts_data=read.csv("dataset.csv")

# Modifying Time Series Data
options(digits.sec=3)
ts_data$date=as.Date(as.character(ts_data$date),format = "%d%B%Y")
oldtime=as.character(ts_data$time)
ts_data$time=strptime(as.character(ts_data$time),"%a %d%B%y %I:%M:%OS %p")
ts_data$time[which(is.na(ts_data$time))]=strptime(oldtime[which(is.na(ts_data$time))],"%a %d%B%y %I:%M:%OS")
ts_data$weekday=strftime(ts_data$time,"%a")
ts_data$timestamp=as.numeric(as.POSIXct(ts_data$time))

# Modifying Static Data
gendatconv=function(x){
  x=as.character(x)
  x1=as.Date(x,"%d%B%Y")
  x2=as.Date(x,"%d-%B-%y")
  x1[which(is.na(x1))]=x2[which(is.na(x1))]
  return(x1)
}
static_data$isin=as.character(static_data$isin)
static_data$ratingAgency1EffectiveDate=gendatconv(static_data$ratingAgency1EffectiveDate)
static_data$ratingAgency2EffectiveDate=gendatconv(static_data$ratingAgency2EffectiveDate)
static_data$issue.date=gendatconv(static_data$issue.date)
static_data$maturity=gendatconv(static_data$maturity)

# Generating Data for Model Training

bts_data=subset(ts_data,ts_data$side=="B")
sts_data=subset(ts_data,ts_data$side=="S")

bts_data$index=paste(bts_data$isin,bts_data$date,sep="_")
bts_price=aggregate(bts_data$price,by=list(bts_data$index),mean)
bts_volume=aggregate(bts_data$volume,by=list(bts_data$index),sum)
bts_data$dummy=1
bts_freq=aggregate(bts_data$dummy,by=list(bts_data$index),sum)
bts_meta=bts_data[!duplicated(bts_data$index),]
bts_meta=bts_meta[order(bts_meta$index),]
bts_price=bts_price[order(bts_price$Group.1),]
bts_freq=bts_freq[order(bts_freq$Group.1),]
bts_volume=bts_volume[order(bts_volume$Group.1),]
buy_data=data.frame(isin=bts_meta$isin,date=bts_meta$date,timestamp=bts_meta$timestamp,
                    volume=bts_volume$x,price=bts_price$x,freq=bts_freq$x,
                    weekday=bts_meta$weekday)
rm(bts_price,bts_freq,bts_data,bts_meta,bts_volume)

sts_data$index=paste(sts_data$isin,sts_data$date,sep="_")
sts_price=aggregate(sts_data$price,by=list(sts_data$index),mean)
sts_volume=aggregate(sts_data$volume,by=list(sts_data$index),sum)
sts_data$dummy=1
sts_freq=aggregate(sts_data$dummy,by=list(sts_data$index),sum)
sts_meta=sts_data[!duplicated(sts_data$index),]
sts_meta=sts_meta[order(sts_meta$index),]
sts_price=sts_price[order(sts_price$Group.1),]
sts_freq=sts_freq[order(sts_freq$Group.1),]
sts_volume=sts_volume[order(sts_volume$Group.1),]
sell_data=data.frame(isin=sts_meta$isin,date=sts_meta$date,timestamp=sts_meta$timestamp,
                     volume=sts_volume$x,price=sts_price$x,freq=sts_freq$x,
                     weekday=sts_meta$weekday)
rm(sts_price,sts_freq,sts_data,sts_meta,sts_volume)

dates=seq(min(ts_data$date),max(ts_data$date),by="days")
weekdays=strftime(dates,"%a")
dates=dates[which(!weekdays %in% c("Sat","Sun"))]
rm(weekdays)

all=expand.grid(date=dates,isin=static_data$isin)
all=all[,c(2:1)]
rm(dates)

all$timestamp=as.numeric(as.POSIXct(all$date))
all$volume=0
all$price=NA
all$freq=0
all$weekday=strftime(all$date,"%a")
all$index=paste(all$isin,all$date,sep="_")

buy_data$index=paste(buy_data$isin,buy_data$date,sep="_")
sell_data$index=paste(sell_data$isin,sell_data$date,sep="_")

allb=subset(all,!(all$index %in% buy_data$index))
alls=subset(all,!(all$index %in% sell_data$index))

buy_data=rbind(buy_data,allb)
sell_data=rbind(sell_data,alls)

buy_data=buy_data[order(buy_data$isin,buy_data$date),]
sell_data=sell_data[order(sell_data$isin,sell_data$date),]

rm(allb,alls,all)

bpricemed=as.data.frame.table(tapply(buy_data$price, buy_data$isin, median, na.rm=TRUE))
library(randomForest)
bpricemed$Freq=na.roughfix(bpricemed$Freq)
buy_data$price[(c(0:17260)*62)+1]=bpricemed$Freq
rm(bpricemed)


spricemed=as.data.frame.table(tapply(sell_data$price, sell_data$isin, median, na.rm=TRUE))
library(randomForest)
spricemed$Freq=na.roughfix(spricemed$Freq)
sell_data$price[(c(0:17260)*62)+1]=spricemed$Freq
rm(spricemed)

library(zoo)

buy_data$price=na.locf(buy_data$price)
sell_data$price=na.locf(sell_data$price)


## Adding Features in static data

buy_data35=subset(buy_data,buy_data$date>"2016-05-05")
sell_data35=subset(sell_data,sell_data$date>"2016-05-05")

static_data=static_data[order(static_data$isin),]
row.names(static_data)=1:nrow(static_data)

buy_avg_vol=tapply(buy_data$volume, buy_data$isin, mean)
buy_avg_vol35=tapply(buy_data35$volume, buy_data35$isin, mean)
buy_var_vol=tapply(buy_data$volume, buy_data$isin, var)
buy_var_vol35=tapply(buy_data35$volume, buy_data35$isin, var)

buy_avg_pri35=tapply(buy_data35$price, buy_data35$isin, mean)
buy_var_pri35=tapply(buy_data35$price, buy_data35$isin, var)

buy_avg_freq=tapply(buy_data$freq, buy_data$isin, mean)
buy_avg_freq35=tapply(buy_data35$freq, buy_data35$isin, mean)

buy_max_vol35=tapply(buy_data35$volume, buy_data35$isin, max)
buy_max_pri=tapply(buy_data$price, buy_data$isin, max)
buy_min_pri=tapply(buy_data$price, buy_data$isin, min)

static_data$buy_avg_vol=as.vector(buy_avg_vol)
static_data$buy_avg_vol35=as.vector(buy_avg_vol35)
static_data$buy_var_vol=as.vector(buy_var_vol)
static_data$buy_var_vol35=as.vector(buy_var_vol35)

static_data$buy_avg_pri35=as.vector(buy_avg_pri35)
static_data$buy_var_pri35=as.vector(buy_var_pri35)

static_data$buy_avg_freq=as.vector(buy_avg_freq)
static_data$buy_avg_freq35=as.vector(buy_avg_freq35)

static_data$buy_max_vol35=as.vector(buy_max_vol35)
static_data$buy_max_pri=as.vector(buy_max_pri)
static_data$buy_min_pri=as.vector(buy_min_pri)

rm(buy_avg_vol,buy_avg_vol35,buy_var_vol,buy_var_vol35,buy_avg_pri35,buy_var_pri35,buy_avg_freq,
   buy_avg_freq35,buy_max_vol35,buy_max_pri,buy_min_pri)

sell_avg_vol=tapply(sell_data$volume, sell_data$isin, mean)
sell_avg_vol35=tapply(sell_data35$volume, sell_data35$isin, mean)
sell_var_vol=tapply(sell_data$volume, sell_data$isin, var)
sell_var_vol35=tapply(sell_data35$volume, sell_data35$isin, var)

sell_avg_pri35=tapply(sell_data35$price, sell_data35$isin, mean)
sell_var_pri35=tapply(sell_data35$price, sell_data35$isin, var)

sell_avg_freq=tapply(sell_data$freq, sell_data$isin, mean)
sell_avg_freq35=tapply(sell_data35$freq, sell_data35$isin, mean)

sell_max_vol35=tapply(sell_data35$volume, sell_data35$isin, max)
sell_max_pri=tapply(sell_data$price, sell_data$isin, max)
sell_min_pri=tapply(sell_data$price, sell_data$isin, min)

static_data$sell_avg_vol=as.vector(sell_avg_vol)
static_data$sell_avg_vol35=as.vector(sell_avg_vol35)
static_data$sell_var_vol=as.vector(sell_var_vol)
static_data$sell_var_vol35=as.vector(sell_var_vol35)

static_data$sell_avg_pri35=as.vector(sell_avg_pri35)
static_data$sell_var_pri35=as.vector(sell_var_pri35)

static_data$sell_avg_freq=as.vector(sell_avg_freq)
static_data$sell_avg_freq35=as.vector(sell_avg_freq35)

static_data$sell_max_vol35=as.vector(sell_max_vol35)
static_data$sell_max_pri=as.vector(sell_max_pri)
static_data$sell_min_pri=as.vector(sell_min_pri)

rm(sell_avg_vol,sell_avg_vol35,sell_var_vol,sell_var_vol35,sell_avg_pri35,sell_var_pri35,sell_avg_freq,
   sell_avg_freq35,sell_max_vol35,sell_max_pri,sell_min_pri)

rm(sell_data35,buy_data35)

## Adding Features in Dynamic Data

buy_run_vol_tot=	c()
buy_run_pri_tot=	c()
buy_run_frq_tot=	c()
buy_run_vol_max_tot=	c()
buy_run_frq_max_tot=	c()
buy_run_pri_min_tot=	c()

sell_run_vol_tot=	c()
sell_run_pri_tot=	c()
sell_run_frq_tot=	c()
sell_run_vol_max_tot=	c()
sell_run_frq_max_tot=	c()
sell_run_pri_min_tot=	c()


for(i in 1:nrow(static_data)){
  buyvol=ts(buy_data$volume[((i-1)*62+1):(62*i)])
  buypri=ts(buy_data$price[((i-1)*62+1):(62*i)])
  buyfrq=ts(buy_data$freq[((i-1)*62+1):(62*i)])
  
  buy_run_vol=cumsum(buyvol)/c(1:62)
  buy_run_pri=cumsum(buypri)/c(1:62)
  buy_run_frq=cumsum(buyfrq)/c(1:62)
  buy_run_vol_max=cummax(buyvol)
  buy_run_frq_max=cummax(buyfrq)
  buy_run_pri_min=cummin(buypri)
  
  sellvol=ts(sell_data$volume[((i-1)*62+1):(62*i)])
  sellpri=ts(sell_data$price[((i-1)*62+1):(62*i)])
  sellfrq=ts(sell_data$freq[((i-1)*62+1):(62*i)])
  
  sell_run_vol=cumsum(sellvol)/c(1:62)
  sell_run_pri=cumsum(sellpri)/c(1:62)
  sell_run_frq=cumsum(sellfrq)/c(1:62)
  sell_run_vol_max=cummax(sellvol)
  sell_run_frq_max=cummax(sellfrq)
  sell_run_pri_min=cummin(sellpri)
  
  buy_run_vol_tot=	c(	buy_run_vol_tot	,	 buy_run_vol	)
  buy_run_pri_tot=	c(	buy_run_pri_tot	,	  buy_run_pri	)
  buy_run_frq_tot=	c(	buy_run_frq_tot	,	  buy_run_frq	)
  buy_run_vol_max_tot=	c(	buy_run_vol_max_tot	,	  buy_run_vol_max	)
  buy_run_frq_max_tot=	c(	buy_run_frq_max_tot	,	  buy_run_frq_max	)
  buy_run_pri_min_tot=	c(	buy_run_pri_min_tot	,	  buy_run_pri_min	)
  sell_run_vol_tot=	c(	sell_run_vol_tot	,	 sell_run_vol	)
  sell_run_pri_tot=	c(	sell_run_pri_tot	,	  sell_run_pri	)
  sell_run_frq_tot=	c(	sell_run_frq_tot	,	  sell_run_frq	)
  sell_run_vol_max_tot=	c(	sell_run_vol_max_tot	,	  sell_run_vol_max	)
  sell_run_frq_max_tot=	c(	sell_run_frq_max_tot	,	  sell_run_frq_max	)
  sell_run_pri_min_tot=	c(	sell_run_pri_min_tot	,	  sell_run_pri_min	)
  
  print(i)
}

buy_data$buy_run_vol_tot=	buy_run_vol_tot
buy_data$buy_run_pri_tot=	buy_run_pri_tot
buy_data$buy_run_frq_tot=	buy_run_frq_tot
buy_data$buy_run_vol_max_tot=	buy_run_vol_max_tot
buy_data$buy_run_frq_max_tot=	buy_run_frq_max_tot
buy_data$buy_run_pri_min_tot=	buy_run_pri_min_tot

buy_data$sell_run_vol_tot=	sell_run_vol_tot
buy_data$sell_run_pri_tot=	sell_run_pri_tot
buy_data$sell_run_frq_tot=	sell_run_frq_tot
buy_data$sell_run_vol_max_tot=	sell_run_vol_max_tot
buy_data$sell_run_frq_max_tot=	sell_run_frq_max_tot
buy_data$sell_run_pri_min_tot=	sell_run_pri_min_tot

sell_data$buy_run_vol_tot=	buy_run_vol_tot
sell_data$buy_run_pri_tot=	buy_run_pri_tot
sell_data$buy_run_frq_tot=	buy_run_frq_tot
sell_data$buy_run_vol_max_tot=	buy_run_vol_max_tot
sell_data$buy_run_frq_max_tot=	buy_run_frq_max_tot
sell_data$buy_run_pri_min_tot=	buy_run_pri_min_tot

sell_data$sell_run_vol_tot=	sell_run_vol_tot
sell_data$sell_run_pri_tot=	sell_run_pri_tot
sell_data$sell_run_frq_tot=	sell_run_frq_tot
sell_data$sell_run_vol_max_tot=	sell_run_vol_max_tot
sell_data$sell_run_frq_max_tot=	sell_run_frq_max_tot
sell_data$sell_run_pri_min_tot=	sell_run_pri_min_tot

rm(buy_run_vol_tot	,
   buy_run_pri_tot	,
   buy_run_frq_tot	,
   buy_run_vol_max_tot	,
   buy_run_frq_max_tot	,
   buy_run_pri_min_tot	,
   sell_run_vol_tot	,
   sell_run_pri_tot	,
   sell_run_frq_tot	,
   sell_run_vol_max_tot	,
   sell_run_frq_max_tot	,
   sell_run_pri_min_tot	,
   buy_run_vol	,
   buy_run_pri	,
   buy_run_frq	,
   buy_run_vol_max	,
   buy_run_frq_max	,
   buy_run_pri_min	,
   sell_run_vol	,
   sell_run_pri	,
   sell_run_frq	,
   sell_run_vol_max	,
   sell_run_frq_max	,
   sell_run_pri_min	
)

rm(buyfrq,buypri,buyvol,sellfrq,sellpri,sellvol,i)

buy_data_train=merge(buy_data,static_data,by="isin",all.x=TRUE)
sell_data_train=merge(sell_data,static_data,by="isin",all.x=TRUE)

buy_data_train$issue.date=as.vector(buy_data_train$date-buy_data_train$issue.date)
buy_data_train$maturity=as.vector(buy_data_train$maturity-buy_data_train$date)
buy_data_train$ratingAgency1EffectiveDate=as.vector(buy_data_train$date-buy_data_train$ratingAgency1EffectiveDate)
buy_data_train$ratingAgency2EffectiveDate=as.vector(buy_data_train$date-buy_data_train$ratingAgency2EffectiveDate)

sell_data_train$issue.date=as.vector(sell_data_train$date-sell_data_train$issue.date)
sell_data_train$maturity=as.vector(sell_data_train$maturity-sell_data_train$date)
sell_data_train$ratingAgency1EffectiveDate=as.vector(sell_data_train$date-sell_data_train$ratingAgency1EffectiveDate)
sell_data_train$ratingAgency2EffectiveDate=as.vector(sell_data_train$date-sell_data_train$ratingAgency2EffectiveDate)

library(randomForest)
sell_data_train$issue.date=na.roughfix(sell_data_train$issue.date)
sell_data_train$couponFrequency=na.roughfix(sell_data_train$couponFrequency)
sell_data_train$maturity=na.roughfix(sell_data_train$maturity)
sell_data_train$ratingAgency1EffectiveDate=na.roughfix(sell_data_train$ratingAgency1EffectiveDate)
sell_data_train$ratingAgency2EffectiveDate=na.roughfix(sell_data_train$ratingAgency2EffectiveDate)

buy_data_train$issue.date=na.roughfix(buy_data_train$issue.date)
buy_data_train$couponFrequency=na.roughfix(buy_data_train$couponFrequency)
buy_data_train$maturity=na.roughfix(buy_data_train$maturity)
buy_data_train$ratingAgency1EffectiveDate=na.roughfix(buy_data_train$ratingAgency1EffectiveDate)
buy_data_train$ratingAgency2EffectiveDate=na.roughfix(buy_data_train$ratingAgency2EffectiveDate)

keep(buy_data_train,sell_data_train,static_data,ts_data,sure=TRUE)

static_data=static_data[order(static_data$isin),]
static_data_temp=static_data

meanbuyvol=as.data.frame.table(tapply(buy_data_train$volume, buy_data_train$isin, mean))
names(meanbuyvol)=c("isin","meanbuyvol")
static_data_temp=merge(static_data_temp,meanbuyvol,by="isin",all.x = TRUE)
rm(meanbuyvol)

meansellvol=as.data.frame.table(tapply(sell_data_train$volume, sell_data_train$isin, mean))
names(meansellvol)=c("isin","meansellvol")
static_data_temp=merge(static_data_temp,meansellvol,by="isin",all.x = TRUE)
rm(meansellvol)

meanbuypri=as.data.frame.table(tapply(buy_data_train$price, buy_data_train$isin, mean))
names(meanbuypri)=c("isin","meanbuypri")
static_data_temp=merge(static_data_temp,meanbuypri,by="isin",all.x = TRUE)
rm(meanbuypri)

meansellpri=as.data.frame.table(tapply(sell_data_train$price, sell_data_train$isin, mean))
names(meansellpri)=c("isin","meansellpri")
static_data_temp=merge(static_data_temp,meansellpri,by="isin",all.x = TRUE)
rm(meansellpri)

varbuyvol=as.data.frame.table(tapply(buy_data_train$volume, buy_data_train$isin, var))
names(varbuyvol)=c("isin","varbuyvol")
static_data_temp=merge(static_data_temp,varbuyvol,by="isin",all.x = TRUE)
rm(varbuyvol)

varsellvol=as.data.frame.table(tapply(sell_data_train$volume, sell_data_train$isin, var))
names(varsellvol)=c("isin","varsellvol")
static_data_temp=merge(static_data_temp,varsellvol,by="isin",all.x = TRUE)
rm(varsellvol)

myfun=function(x){
  length(which(x==0))
}

zerobuyvol=as.data.frame.table(tapply(buy_data_train$volume, buy_data_train$isin, myfun))
names(zerobuyvol)=c("isin","zerobuyvol")
static_data_temp=merge(static_data_temp,zerobuyvol,by="isin",all.x = TRUE)
rm(zerobuyvol)

zerosellvol=as.data.frame.table(tapply(sell_data_train$volume, sell_data_train$isin, myfun))
names(zerosellvol)=c("isin","zerosellvol")
static_data_temp=merge(static_data_temp,zerosellvol,by="isin",all.x = TRUE)
rm(zerosellvol)

buy_data_train2=subset(buy_data_train,buy_data_train$date>="2016-06-01")
zerofbuyvol=as.data.frame.table(tapply(buy_data_train2$volume, buy_data_train2$isin, myfun))
names(zerofbuyvol)=c("isin","zerofbuyvol")
static_data_temp=merge(static_data_temp,zerofbuyvol,by="isin",all.x = TRUE)
rm(zerofbuyvol,buy_data_train2)

sell_data_train2=subset(sell_data_train,sell_data_train$date>="2016-06-01")
zerofsellvol=as.data.frame.table(tapply(sell_data_train2$volume, sell_data_train2$isin, myfun))
names(zerofsellvol)=c("isin","zerofsellvol")
static_data_temp=merge(static_data_temp,zerofsellvol,by="isin",all.x = TRUE)
rm(zerofsellvol,sell_data_train2)

rm(myfun)
static_data=static_data[,1:24]

##############################################################################################

## Baseline Approach : Independent Mean of Bond Liquidity

ts_data=ts_data[order(ts_data$isin,ts_data$timestamp),]

count=as.data.frame.table(table(ts_data$isin))
count=count[order(-count$Freq),]

rm(count)

days=86
bts_data=subset(ts_data,ts_data$side=="B")
bd=as.data.frame.table(tapply(bts_data$volume/days, bts_data$isin, sum))
rm(bts_data)
sts_data=subset(ts_data,ts_data$side=="S")
sd=as.data.frame.table(tapply(sts_data$volume/days, sts_data$isin, sum))
rm(sts_data)

bsd=merge(bd,sd,by="Var1")
names(bsd)=names(output)

bsd[which(is.na(bsd$buyvolume)),"buyvolume"]=bsd[which(is.na(bsd$buyvolume)),"sellvolume"]
bsd[which(is.na(bsd$sellvolume)),"sellvolume"]=bsd[which(is.na(bsd$sellvolume)),"buyvolume"]

bsd$buyvolume=bsd$buyvolume*3
bsd$sellvolume=bsd$sellvolume*3

output00=bsd

rm(bsd,bd,sd,days)

##############################################################################################

## Approach 1 : Independent Statistical Analysis of Time Series Data

# Windwoing Mean

ts_data$isin=as.factor(ts_data$isin)
ts_data=ts_data[order(ts_data$isin,ts_data$timestamp),]

count=as.data.frame.table(table(ts_data$isin))
count=count[order(-count$Freq),]

rm(count)

output=data.frame(isin=static_data$isin,buyvolume=NA, sellvolume=NA)

days=35
bts_data=subset(ts_data,ts_data$side=="B" & ts_data$date>"2016-05-06")
bd=as.data.frame.table(tapply(bts_data$volume/days, bts_data$isin, sum))
rm(bts_data)
sts_data=subset(ts_data,ts_data$side=="S" & ts_data$date>"2016-05-06")
sd=as.data.frame.table(tapply(sts_data$volume/days, sts_data$isin, sum))
rm(sts_data)

bsd=merge(bd,sd,by="Var1")
names(bsd)=names(output)

bsd[which(is.na(bsd$buyvolume)),"buyvolume"]=0
bsd[which(is.na(bsd$sellvolume)),"sellvolume"]=0

bsd$buyvolume=bsd$buyvolume*2.7
bsd$sellvolume=bsd$sellvolume*2.7

bsd$buyvolume=ifelse(bsd$buyvolume>=0,bsd$buyvolume,0)
bsd$sellvolume=ifelse(bsd$sellvolume>=0,bsd$sellvolume,0)

output11=bsd
rm(bsd,bd,sd,days,output)

# Mean + Variance

buy_data_traint=subset(buy_data_train,buy_data_train$date>"2016-03-15")
sell_data_traint=subset(sell_data_train,sell_data_train$date>"2016-03-15")

output=data.frame(isin=unique(buy_data_traint$isin))

myfun=function(x){
  alpha=0.05
  return(mean(x)-alpha*sqrt(var(x)))
}

buy_mean=as.data.frame.table(tapply(buy_data_traint$volume, buy_data_traint$isin, mean))
sell_mean=as.data.frame.table(tapply(sell_data_traint$volume, sell_data_traint$isin, mean))

output=merge(output,buy_mean,by.x = "isin",by.y = "Var1",all.x = TRUE)
output=merge(output,sell_mean,by.x = "isin",by.y = "Var1",all.x = TRUE)

names(output)=c("isin","buyvolume","sellvolume")

output$buyvolume=ifelse(output$buyvolume>=0,output$buyvolume,0)
output$sellvolume=ifelse(output$sellvolume>=0,output$sellvolume,0)

output$buyvolume=output$buyvolume*1.5
output$sellvolume=output$sellvolume*1.5

output12=output
rm(buy_mean,sell_mean,output,buy_data_traint,sell_data_traint,output,myfun)


##############################################################################################

## Approach 2 : Independent Time Series Forecasting

# ETS on all 62 working days data

library(forecast)
output=data.frame(isin=unique(buy_data_train$isin),buyvolume=NA, sellvolume=NA)
for(i in 1:nrow(output)){
  sbst=buy_data_train$volume[((i-1)*62+1):(i*62)]
  myts=ts(sbst)
  fit=ets(myts)
  out=forecast(fit,3)
  output$buyvolume[i]=max(0,sum(out$mean))
  sbst=sell_data_train$volume[((i-1)*62+1):(i*62)]
  myts=ts(sbst)
  fit=ets(myts)
  out=forecast(fit,3)
  output$sellvolume[i]=max(0,sum(out$mean))
  print(i)
}

output$buyvolume=output$buyvolume*0.7
output$sellvolume=output$sellvolume*0.7
output21=output
rm(sbst,output,myts,fit,out,i)

# ARIMA on all 62 working days data

library(forecast)
output=data.frame(isin=unique(buy_data_train$isin),buyvolume=NA, sellvolume=NA)
for(i in 1:nrow(output)){
  sbst=buy_data_train$volume[((i-1)*62+1):(i*62)]
  myts=ts(sbst)
  fit=auto.arima(myts)
  out=forecast(fit,3)
  output$buyvolume[i]=max(0,sum(out$mean))
  sbst=sell_data_train$volume[((i-1)*62+1):(i*62)]
  myts=ts(sbst)
  fit=auto.arima(myts)
  out=forecast(fit,3)
  output$sellvolume[i]=max(0,sum(out$mean))
  print(i)
}

output$buyvolume=output$buyvolume*0.7
output$sellvolume=output$sellvolume*0.7

output22=output
rm(sbst,output,myts,fit,out,i)

# ETS on last 35 working days data

buy_data_train_temp=subset(buy_data_train,buy_data_train$date>"2016-05-02")
sell_data_train_temp=subset(sell_data_train,sell_data_train$date>"2016-05-02")

row.names(buy_data_train_temp)=1:nrow(buy_data_train_temp)
row.names(sell_data_train_temp)=1:nrow(sell_data_train_temp)

library(forecast)
output=data.frame(isin=unique(buy_data_traintemp$isin),buyvolume=NA, sellvolume=NA)
for(i in 1:nrow(output)){
  sbst=buy_data_traintemp$volume[((i-1)*62+1):(i*62)]
  myts=ts(sbst)
  fit=ets(myts)
  out=forecast(fit,3)
  output$buyvolume[i]=max(0,sum(out$mean))
  sbst=sell_data_traintemp$volume[((i-1)*62+1):(i*62)]
  myts=ts(sbst)
  fit=ets(myts)
  out=forecast(fit,3)
  output$sellvolume[i]=max(0,sum(out$mean))
  print(i)
}

output$buyvolume=output$buyvolume*0.6
output$sellvolume=output$sellvolume*0.6
output23=output
rm(sbst,output,myts,fit,out,i)


# ARIMA on last 35 working days data


library(forecast)
output=data.frame(isin=unique(buy_data_traintemp$isin),buyvolume=NA, sellvolume=NA)
for(i in 1:nrow(output)){
  sbst=buy_data_traintemp$volume[((i-1)*62+1):(i*62)]
  myts=ts(sbst)
  fit=auto.arima(myts)
  out=forecast(fit,3)
  output$buyvolume[i]=max(0,sum(out$mean))
  sbst=sell_data_traintemp$volume[((i-1)*62+1):(i*62)]
  myts=ts(sbst)
  fit=auto.arima(myts)
  out=forecast(fit,3)
  output$sellvolume[i]=max(0,sum(out$mean))
  print(i)
}

output$buyvolume=output$buyvolume*0.6
output$sellvolume=output$sellvolume*0.6

output24=output
rm(sbst,output,myts,fit,out,i,buy_data_train_temp,sell_data_train_temp)

##############################################################################################

## Ensembling Best of Approach 1 and Approach 2
out1=output11
out2=output21

# Tuned based on cross validation results
alpha1=0.6
alpha2=0.6
out=out1
out$buyvolume=alpha1*out1$buyvolume + (1-alpha1)*out2$buyvolume
out$sellvolume=alpha2*out1$sellvolume + (1-alpha2)*out2$sellvolume


output25=out
rm(out,out1,out2,alpha1,alpha2)


##############################################################################################

## Approach 3 : Renormalization using Correlation

# Static Data Correlation
static_numeric=static_data[,c(3,5,6,8,9,14,21,24,25,26,27,28,31,32)]
static_factor=static_data[,c(4,10,12,15,16,17,18,20,23)]

static_numeric$issue.date=as.vector(static_numeric$issue.date-mean(static_numeric$issue.date,na.rm=TRUE))
static_numeric$maturity=as.vector(static_numeric$maturity-mean(static_numeric$maturity,na.rm=TRUE))
static_numeric$ratingAgency1EffectiveDate=as.vector(static_numeric$ratingAgency1EffectiveDate-mean(static_numeric$ratingAgency1EffectiveDate,na.rm=TRUE))
static_numeric$ratingAgency2EffectiveDate=as.vector(static_numeric$ratingAgency2EffectiveDate-mean(static_numeric$ratingAgency2EffectiveDate,na.rm=TRUE))
library(randomForest)
static_numeric=na.roughfix(static_numeric)

normalizer=function(df){
  for(i in 1:ncol(df)){
    if((quantile(df[,i])[4]-quantile(df[,i])[2])!=0)
      df[,i]=(df[,i]-mean(df[,i]))/(quantile(df[,i])[4]-quantile(df[,i])[2])
    else
      df[,i]=(df[,i]-mean(df[,i]))/(max(df[,i])-min(df[,i]))
  }
  return(df)
}
static_numeric=normalizer(static_numeric)

static_numeric=as.matrix(t(static_numeric))
static_numeric_cor=cor(static_numeric)
rm(static_numeric)

dummycreator=function(df){
  for(i in 1:ncol(df)){
    a=nlevels(df[,i])
    ndf=as.data.frame(matrix(NA,ncol = a,nrow=nrow(df)))
    b=levels(df[,i])
    names(ndf)=paste(names(df)[i],b,sep="")
    for(j in 1:a){
      ndf[,j]=ifelse(df[,i]==b[j],1,0)
    }
    df=cbind(df,ndf)
  }
  return(df)
}
static_factor=dummycreator(static_factor)
static_factor=static_factor[,-c(1:9)]
static_factor_temp=static_factor
static_factor=as.matrix(t(static_factor))
static_factor_cor=cor(static_factor)
rm(static_factor)

static_cor=(static_numeric_cor+static_factor_cor)/2

rm(static_numeric_cor,static_factor_cor,normalizer,dummycreator)

# Dynamic Data Correlation

buydynmat=matrix(data=NA,nrow=62,ncol = nrow(static_data))
for(i in 1:ncol(buydynmat)){
  buydynmat[,i]=buy_data_train$volume[((i-1)*62+1):(i*62)]
  print(i)
}
rm(i)

selldynmat=matrix(data=NA,nrow=62,ncol = nrow(static_data))
for(i in 1:ncol(selldynmat)){
  selldynmat[,i]=sell_data_train$volume[((i-1)*62+1):(i*62)]
  print(i)
}
rm(i)

dynbuycor=cor(buydynmat)
dynsellcor=cor(selldynmat)
rm(buydynmat,selldynmat)

dynbuycor[is.na(dynbuycor)]=0
dynsellcor[is.na(dynsellcor)]=0
dyncor=(dynbuycor+dynsellcor)/2

rm(dynbuycor,dynsellcor)

#Taking Best Output
output=output25

cor=(dyncor+static_cor)/2
cor[cor<=0.5]=0
cor=cor*cor
output2=output
for(i in 1:nrow(output)){
  output2$buyvolume[i]=sum(output2$buyvolume*cor[,i])/sum(cor[,i])
  output2$sellvolume[i]=sum(output2$sellvolume*cor[,i])/sum(cor[,i])
  print(i)
}

output31=output2
rm(output2,i)

dyncor=dyncor*dyncor
output3=output
for(i in 1:nrow(output)){
  output3$buyvolume[i]=sum(output3$buyvolume*dyncor[,i])/sum(dyncor[,i])
  output3$sellvolume[i]=sum(output3$sellvolume*dyncor[,i])/sum(dyncor[,i])
  print(i)
}

output32=output3
rm(output3,i)
rm(dyncor,static_cor,cor)

##############################################################################################

## Approach 4 : Renormalization using Grouping & Hierarchical Clustering

distan=dist(static_factor_temp,method = "euclidean")
clusters=hclust(cor,method = "ward.D")
clusters=cutree(hclust,k=30)
static_data$clustergroup=as.factor(clusters)
rm(distan,clusters)

ts_data$isin=as.factor(ts_data$isin)
ts_data=ts_data[order(ts_data$isin,ts_data$timestamp),]

output=data.frame(isin=static_data$isin,buyvolume=NA, sellvolume=NA)

# Hyperparams tuned using Cross Validation
days=35
alpha=0.9
beta=0.9

bts_data=subset(ts_data,ts_data$side=="B" & ts_data$date>"2016-05-06")
bd=as.data.frame.table(tapply(bts_data$volume/days, bts_data$isin, sum))
bts_data=merge(bts_data,static_data[,c("isin","clustergroup")],by="isin",all.x = TRUE)
bd2=as.data.frame.table(tapply(bts_data$volume/days, bts_data$clustergroup, sum ))
temp=as.data.frame.table(table(static_data$clustergroup))
bd2$Freq=bd2$Freq/temp$Freq
rm(bts_data)
bd=bd[order(bd$Var1),]
static_data=static_data[order(static_data$isin),]
bd$Group=static_data$clustergroup
bd=merge(bd,bd2,by.x = "Group",by.y = "Var1",all.x = TRUE)
bd$Group=NULL
names(bd)=c("isin","buyvolume","buyvolumeavg")
rm(bd2,temp)
bd[which(is.na(bd$buyvolume)),"buyvolume"]=0
bd$buyvol=alpha*bd$buyvolume+(1-alpha)*bd$buyvolumeavg
bd=bd[order(bd$isin),]

sts_data=subset(ts_data,ts_data$side=="S" & ts_data$date>"2016-05-06")
sd=as.data.frame.table(tapply(sts_data$volume/days, sts_data$isin, sum))
sts_data=merge(sts_data,static_data[,c("isin","clustergroup")],by="isin",all.x = TRUE)
sd2=as.data.frame.table(tapply(sts_data$volume/days, sts_data$clustergroup, sum ))
temp=as.data.frame.table(table(static_data$clustergroup))
sd2$Freq=sd2$Freq/temp$Freq
rm(sts_data)
sd=sd[order(sd$Var1),]
static_data=static_data[order(static_data$isin),]
sd$Group=static_data$clustergroup
sd=merge(sd,sd2,by.x = "Group",by.y = "Var1",all.x = TRUE)
sd$Group=NULL
names(sd)=c("isin","sellvolume","sellvolumeavg")
rm(sd2,temp)
sd[which(is.na(sd$sellvolume)),"sellvolume"]=0
sd$sellvol=beta*sd$sellvolume+(1-beta)*sd$sellvolumeavg
sd=sd[order(sd$isin),]

bsd=merge(bd[,c("isin","buyvol")],sd[,c("isin","sellvol")],by="isin")
names(bsd)=c("isin","buyvolume","sellvolume")

bsd$buyvolume=ifelse(bsd$buyvolume>=0,bsd$buyvolume,0)
bsd$sellvolume=ifelse(bsd$sellvolume>=0,bsd$sellvolume,0)

output41=bsd
rm(bsd,sd,bd)

##############################################################################################

## Approach 5 : Regression

#Preparing Test Data
date=as.Date(c("2016-06-10","2016-06-13","2016-06-14"))
isin=as.character(unique(buy_data_train$isin))

test_data=expand.grid(date=date,isin=isin)
test_data=test_data[,c(2:1)]
rm(date,isin)

buy_data_temp=subset(buy_data_train,buy_data_train$date>"2016-06-06")
sell_data_temp=subset(buy_data_train,buy_data_train$date>"2016-06-06")

buy_test_data=cbind(test_data,buy_data_temp[,-c(1,2)])
sell_test_data=cbind(test_data,sell_data_temp[,-c(1,2)])

rm(buy_data_temp,sell_data_temp,test_data)

buy_test_data$timestamp=as.numeric(as.POSIXct(buy_test_data$date))
buy_test_data$weekday=as.factor(strftime(buy_test_data$date,"%a"))
buy_test_data$index=paste(buy_test_data$isin,buy_test_data$date)

output2=output25
output2$buyvolume=ifelse(static_data_temp$zerofbuyvol==0,output2$buyvolume*1.1,output2$buyvolume*0.95)
output2$sellvolume=ifelse(static_data_temp$zerofbuyvol==0,output2$sellvolume*1.1,output2$sellvolume*0.95)

output71=output2
rm(output2)

output2=output71
output2$buyvolume=output71$buyvolume*0.5 + output71$sellvolume*0.5
output2$sellvolume=output71$buyvolume*0.5 + output71$sellvolume*0.5
output73=output2
rm(output2)

sell_test_data$timestamp=as.numeric(as.POSIXct(sell_test_data$date))
sell_test_data$weekday=as.factor(strftime(sell_test_data$date,"%a"))
sell_test_data$index=paste(sell_test_data$isin,sell_test_data$date)

# XGBOOST Regression Model
library(caret)
library(doParallel)

trctrl=trainControl(method="repeatedcv",number = 5,repeats=1)

cl=makeCluster(6)
registerDoParallel(cl)

buy_trained=train(volume~.,
                  data=buy_data_train[,-c(1,2,3,8,21,26,30,31,32,36)],
                  method="xgbTree",
                  trControl=trctrl)

stopCluster(cl)

cl=makeCluster(6)
registerDoParallel(cl)

output2=output73
output2$buyvolume=output2$buyvolume*0.95
output2$sellvolume=output2$sellvolume*1
output73=output2
rm(output2)

sell_trained=train(volume~.,
                   data=buy_data_train[,-c(1,2,3,8,21,26,30,31,32,36)],
                   method="xgbTree",
                   trControl=trctrl)

stopCluster(cl)

output2=output73
output2$buyvolume=ifelse(static_data$Market=="Market0",output2$buyvolume*0.85,output2$buyvolume*1)
output2$sellvolume=ifelse(static_data$Market=="Market0",output2$sellvolume*1,output2$sellvolume*1)
output73=output2
rm(output2)

buy_test_data$volume=predict(buy_trained,newdata=buy_test_data)
sell_test_data$volume=predict(sell_trained,newdata=sell_test_data)

buy_vol=as.data.frame.table(tapply(buy_test_data$volume, buy_test_data$isin, sum))
sell_vol=as.data.frame.table(tapply(sell_test_data$volume, sell_test_data$isin, sum))

output51=data.frame(isin=buy_vol$Var1,buyvolume=buy_vol$Freq,sellvolume=sell_vol$Freq)
rm(buy_trained,sell_trained,cl,trctrl,buy_vol,sell_vol)

# Linear  Regression  Model

trctrl=trainControl(method="repeatedcv",number = 5,repeats=1)

cl=makeCluster(6)
registerDoParallel(cl)

output2=output73
output2$buyvolume=ifelse(static_data$couponType=="couponType1",output2$buyvolume*1.1,output2$buyvolume*1)
output2$buyvolume=ifelse(static_data$couponType=="couponType7",output2$buyvolume*1.3,output2$buyvolume*1)
output2$sellvolume=ifelse(static_data$couponType=="couponType1",output2$sellvolume,output2$sellvolume*1)
output63=output2
rm(output2)

buy_trained=train(volume~.,
                  data=buy_data_train[,-c(1,2,3,8,21,26,30,31,32,36)],
                  method="lm",
                  trControl=trctrl)

stopCluster(cl)

cl=makeCluster(6)
registerDoParallel(cl)

sell_trained=train(volume~.,
                   data=buy_data_train[,-c(1,2,3,8,21,26,30,31,32,36)],
                   method="lm",
                   trControl=trctrl)

stopCluster(cl)

buy_test_data$volume=predict(buy_trained,newdata=buy_test_data)
sell_test_data$volume=predict(sell_trained,newdata=sell_test_data)

buy_vol=as.data.frame.table(tapply(buy_test_data$volume, buy_test_data$isin, sum))
sell_vol=as.data.frame.table(tapply(sell_test_data$volume, sell_test_data$isin, sum))

output52=data.frame(isin=buy_vol$Var1,buyvolume=buy_vol$Freq,sellvolume=sell_vol$Freq)
rm(buy_trained,sell_trained,cl,trctrl,buy_vol,sell_vol)

# Random Forest Regression Model

trctrl=trainControl(method="repeatedcv",number = 5,repeats=1)

cl=makeCluster(6)
registerDoParallel(cl)

buy_trained=train(volume~.,
                  data=buy_data_train[,-c(1,2,3,8,21,26,30,31,32,36)],
                  method="rf",
                  trControl=trctrl)

stopCluster(cl)

cl=makeCluster(6)
registerDoParallel(cl)

sell_trained=train(volume~.,
                   data=buy_data_train[,-c(1,2,3,8,21,26,30,31,32,36)],
                   method="rf",
                   trControl=trctrl)

stopCluster(cl)

buy_test_data$volume=predict(buy_trained,newdata=buy_test_data)
sell_test_data$volume=predict(sell_trained,newdata=sell_test_data)

buy_vol=as.data.frame.table(tapply(buy_test_data$volume, buy_test_data$isin, sum))
sell_vol=as.data.frame.table(tapply(sell_test_data$volume, sell_test_data$isin, sum))

output53=data.frame(isin=buy_vol$Var1,buyvolume=buy_vol$Freq,sellvolume=sell_vol$Freq)
rm(buy_trained,sell_trained,cl,trctrl,buy_vol,sell_vol)

##############################################################################################

## Modifications 1 : Ensembling Various Approaches

out1=output51
out2=output25

# Tuned based on cross validation results
alpha1=0.5
alpha2=0.5
out=out1
out$buyvolume=alpha1*out1$buyvolume + (1-alpha1)*out2$buyvolume
out$sellvolume=alpha2*out1$sellvolume + (1-alpha2)*out2$sellvolume


output61=out
rm(out,out1,out2,alpha1,alpha2)

out1=output53
out2=output25

# Tuned based on cross validation results
alpha1=0.5
alpha2=0.5
out=out1
out$buyvolume=alpha1*out1$buyvolume + (1-alpha1)*out2$buyvolume
out$sellvolume=alpha2*out1$sellvolume + (1-alpha2)*out2$sellvolume


output62=out
rm(out,out1,out2,alpha1,alpha2)


##############################################################################################

## Modifications 2 : Heuristics based on Domain Knowledge

# Bonds which are being traded continuously for past 10 days is more likely to be traded next day

output2=output25
output2$buyvolume=ifelse(static_data_temp$zerofbuyvol==0,output2$buyvolume*1.1,output2$buyvolume*0.95)
output2$sellvolume=ifelse(static_data_temp$zerofbuyvol==0,output2$sellvolume*1.1,output2$sellvolume*0.95)

output71=output2
rm(output2)

# Bonds which have been issued recently would have spikes

output=output25

summary(static_data$issue.date)
issuedrecently=static_data$isin[which(static_data$issue.date<"2016-06-18" & static_data$issue.date>"2016-06-01")]
buy_data_train_sub=subset(buy_data_train,buy_data_train$isin %in% issuedrecently 
                          & buy_data_train$date>"2016-06-01")
sell_data_train_sub=subset(sell_data_train,sell_data_train$isin %in% issuedrecently 
                           & sell_data_train$date>"2016-06-01")
static_data_sub=subset(static_data,static_data$isin %in% issuedrecently)
buy_data_train_sub$isin=as.character(buy_data_train_sub$isin)
sell_data_train_sub$isin=as.character(sell_data_train_sub$isin)
buy_data_train_sub=as.data.frame.table(tapply(buy_data_train_sub$volume, buy_data_train_sub$isin, mean))
sell_data_train_sub=as.data.frame.table(tapply(sell_data_train_sub$volume, sell_data_train_sub$isin, mean))

outputbefore=merge(buy_data_train_sub,sell_data_train_sub,by="Var1")
names(outputbefore)=c("isin","buyvolume","sellvolume")
output=output[-which(output$isin %in% outputbefore$isin),]
output=rbind(output,outputbefore)
output=output[order(output$isin),]

rm(static_data_sub,buy_data_train_sub,sell_data_train_sub,outputbefore,issuedrecently)
output72=output
rm(output)

# Selling and Buying Volumes are usually equal

output2=output71
output2$buyvolume=output71$buyvolume*0.5 + output71$sellvolume*0.5
output2$sellvolume=output71$buyvolume*0.5 + output71$sellvolume*0.5
output73=output2

rm(output2)


# Some Maturity type coupons have excess predicted values

tapply(static_data$meanbuyvol, static_data$maturityType, mean)
output2=output63
output2$buyvolume=ifelse(static_data$maturityType=="maturityType5",output2$buyvolume*0.9,output2$buyvolume*1)
output2$buyvolume=ifelse(static_data$maturityType=="maturityType10",output2$buyvolume*0.8,output2$buyvolume*1)
output2$sellvolume=ifelse(static_data$maturityType=="maturityType5",output2$sellvolume*0.9,output2$sellvolume*1)
output2$sellvolume=ifelse(static_data$maturityType=="maturityType10",output2$sellvolume*0.8,output2$sellvolume*1)
#Rest of them increased by a constant factor
output2$sellvolume=output2$sellvolume*1.1
output63=output2
rm(output2)

##############################################################################################

## Final Output File

write.csv(output63,"output.csv",row.names = FALSE,quote = FALSE)

##############################################################################################
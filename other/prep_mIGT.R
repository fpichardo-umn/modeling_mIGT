rm(list=ls())
require("foreign")


#### load trial-level data ####

wave1.raw<-read.spss("modigt_data_Wave1.sav",to.data.frame = TRUE)
wave2.raw<-read.spss("modigt_data_Wave2.sav",to.data.frame = TRUE)
wave3.raw<-read.spss("modigt_data_Wave3.sav",to.data.frame = TRUE)

wave1.raw$wave<-"w1"
wave2.raw$wave<-"w2"
wave3.raw$wave<-"w3"


write.csv(wave1.raw,file="wave_1_mIGT.csv",row.names = F)

# if there are multiple administrations, label them

gng.wave1.raw$admin<-NA

for (s in unique(gng.wave1.raw$sid)){
  sts<-unique(gng.wave1.raw[gng.wave1.raw$sid==s,]$s_starttime)
  adm<-paste0("a",1:length(sts))
  
  for (a in 1:length(sts)){
    gng.wave1.raw[gng.wave1.raw$sid==s & 
                    gng.wave1.raw$s_starttime==sts[a],]$admin<-adm[a]
  }
}

gng.wave2.raw$admin<-NA

for (s in unique(gng.wave2.raw$sid)){
  sts<-unique(gng.wave2.raw[gng.wave2.raw$sid==s,]$s_starttime)
  adm<-paste0("a",1:length(sts))
  
  for (a in 1:length(sts)){
    gng.wave2.raw[gng.wave2.raw$sid==s & 
                    gng.wave2.raw$s_starttime==sts[a],]$admin<-adm[a]
  }
}

gng.wave3.raw$admin<-NA

for (s in unique(gng.wave3.raw$sid)){
  sts<-unique(gng.wave3.raw[gng.wave3.raw$sid==s,]$s_starttime)
  adm<-paste0("a",1:length(sts))
  
  for (a in 1:length(sts)){
    gng.wave3.raw[gng.wave3.raw$sid==s & 
                    gng.wave3.raw$s_starttime==sts[a],]$admin<-adm[a]
  }
}

#save.image("raw_gng_processed.RData")

needed<-c("sid","wave","stimulusitem2","response","responsetype","latency","admin","s_starttime","s_startdate")

gng.raw<-rbind(gng.wave1.raw[,needed],
               gng.wave2.raw[,needed])

gng.raw<-rbind(gng.raw,
               gng.wave3.raw[,needed])


# trim non-probes
gng.raw$stimulusitem2<-gsub(" ","",as.character(gng.raw$stimulusitem2))
gng.raw<-gng.raw[gng.raw$stimulusitem2!="<PressEntertocontinue>",]
gng.raw<-gng.raw[gng.raw$stimulusitem2!=0,]

# remove atypical responses
gng.raw$response<-gsub(" ","",gng.raw$response)
gng.raw<-gng.raw[gng.raw$response%in%c("57","0"),]

#### format relevant columns for checks and model fitting ####

gng.raw$s<-paste0(gng.raw$sid,gng.raw$wave,gng.raw$admin)

# stimulus

gng.raw$S<-NA
gng.raw[gng.raw$stimulusitem2!="X",]$S<-"go"
gng.raw[gng.raw$stimulusitem2=="X",]$S<-"ng"
gng.raw$S<-factor(gng.raw$S,levels = c("ng","go"))

# response
gng.raw$R<-NA
gng.raw[gng.raw$response=="0",]$R<-"TO"
gng.raw[gng.raw$response=="57",]$R<-"RESP"
gng.raw$R<-as.factor(gng.raw$R)


# response time (RT)
gng.raw$RT<-NA
gng.raw[gng.raw$R=="RESP",]$RT<-gng.raw[gng.raw$R=="RESP",]$latency/1000
gng.raw$RT<-round(gng.raw$RT,3)


plot(density(gng.raw$RT,na.rm = TRUE),col="green")


max(gng.raw$RT,na.rm = TRUE)
#[1] 4.005


#### checks and possible exclusion criteria #######


# exclude fast guesses and abnormally long RT from GNG data

gng.dat<-gng.raw[gng.raw$RT>=.150 | is.na(gng.raw$RT) ,]
gng.dat<-gng.dat[gng.dat$RT<=3.000 | is.na(gng.dat$RT),]
# excludes ~1% of trials
1-(length(gng.dat$RT)/length(gng.raw$RT))

# specify summary data frame variables 
# subject number
sum.stats<-data.frame(ID=unique(c(gng.raw$s)))

#stats for beh sessions

sum.stats$sid<-NA
sum.stats$wave<-NA
sum.stats$admin<-NA
sum.stats$s_startdate<-NA
sum.stats$s_starttime<-NA
sum.stats$gng.total.acc<-NA # total accuracy
sum.stats$gng.total.n<-NA # number of trials availabe to model
sum.stats$gng.fast.g<-NA # number of "fast guess" trials excluded
sum.stats$gng.FA.g<-NA # number of false alarms
sum.stats$gng.OM.g<-NA # number of omissions
sum.stats$gng.go_SDRT<-NA # go RT standard deviation
sum.stats$gng.go_MRT<-NA # go RT mean 
sum.stats$gng.go_acc<-NA # go accuracy
sum.stats$gng.nogo_SDRT<-NA # nogo RT standard deviation
sum.stats$gng.nogo_MRT<-NA # nogo RT mean 
sum.stats$gng.nogo_acc<-NA # nogo accuracy

# calculate all summary variables for each person
for (s in sum.stats$ID){
  if(length(gng.raw[gng.raw$s==s,]$RT)>1){
    tmp<-gng.dat[gng.dat$s==s,]
    sum.stats[sum.stats$ID==s,]$sid<-tmp$sid[1]
    sum.stats[sum.stats$ID==s,]$wave<-tmp$wave[1]
    sum.stats[sum.stats$ID==s,]$admin<-tmp$admin[1]
    sum.stats[sum.stats$ID==s,]$s_startdate<-tmp$s_startdate[1]
    sum.stats[sum.stats$ID==s,]$s_starttime<-tmp$s_starttime[1]
    sum.stats[sum.stats$ID==s,]$gng.total.acc<-(length(tmp[tmp$S=="go" & tmp$R=="RESP",]$RT)+
                                                  length(tmp[tmp$S=="ng" & tmp$R=="TO",]$RT))/length(tmp$RT) 
    sum.stats[sum.stats$ID==s,]$gng.total.n<-length(tmp$RT) 
    sum.stats[sum.stats$ID==s,]$gng.fast.g<-(length(gng.raw[gng.raw$s==s,]$RT)-length(tmp$RT))
    sum.stats[sum.stats$ID==s,]$gng.FA.g<-length(tmp[tmp$S=="ng" & tmp$R=="RESP",]$RT)
    sum.stats[sum.stats$ID==s,]$gng.OM.g<-length(tmp[tmp$S=="go" & tmp$R=="TO",]$RT)
    sum.stats[sum.stats$ID==s,]$gng.go_SDRT<-sd(tmp[tmp$S=="go" & tmp$R=="RESP",]$RT,na.rm = TRUE)
    sum.stats[sum.stats$ID==s,]$gng.go_MRT<-mean(tmp[tmp$S=="go" & tmp$R=="RESP",]$RT,na.rm = TRUE)
    sum.stats[sum.stats$ID==s,]$gng.go_acc<-length(tmp[tmp$S=="go" & tmp$R=="RESP",]$RT)/length(tmp[tmp$S=="go",]$RT)
    sum.stats[sum.stats$ID==s,]$gng.nogo_SDRT<-sd(tmp[tmp$S=="ng" & tmp$R=="RESP",]$RT,na.rm = TRUE)
    sum.stats[sum.stats$ID==s,]$gng.nogo_MRT<-mean(tmp[tmp$S=="ng" & tmp$R=="RESP",]$RT,na.rm = TRUE)
    sum.stats[sum.stats$ID==s,]$gng.nogo_acc<-length(tmp[tmp$S=="ng" & tmp$R=="TO",]$RT)/length(tmp[tmp$S=="ng",]$RT)
  }
}

save.image("raw_gng_processed.RData")


# exclude sessions with <.55 accuracy or >10% fast-guesses and
# <200 trials after  exclusions 

sum.stats$inc<-(sum.stats$gng.total.acc>=.55 & sum.stats$gng.total.n>=200 & sum.stats$gng.fast.g<25)

# START HERE!
mean(sum.stats$inc)*100
#[1] 78.39977

# save out summary file 
write.csv(sum.stats,"ahrbgng_sum_stats.csv",row.names = FALSE)

#### save out DMC data for included sessions

gng.DMC<-gng.dat[gng.dat$s%in%sum.stats[sum.stats$inc,]$ID,]
gng.DMC<-gng.DMC[,c("s","S","R","RT")]
gng.DMC$s<-as.factor(as.character(gng.DMC$s))

save(gng.DMC,file="ahrb_gng_data_dmc.RData")

###########
# Demographics and other variables

tmp<-read.spss("gonogo_summary_Wave1.sav",to.data.frame = TRUE)

### ===================================================================================================
### Prototype R code
### "A real-time data assimilative forecasting system for animal tracking"
### by M. Randon, M. Dowd and R. Joy
### ===================================================================================================

# The code implements a real-time data assimilative forecasting system 
# for estimating current and future animal locations.
# We first simulate the data: 
# (1) a preferred habitat using a potential function approach
# (2) a track of visual observations based on a realization of the 
# continuous-time correlated random walk. 
# We then run the data assimilation cycle with a particle filter algorithm. 
# State augmentation is used for online estimation of the persistence parameter (phi).
# A drift term is computed using the gradient of a potential function to steer the animal movement.
# Finally, we provide some basic visualizations of the outputs.

rm(list=ls())

library(MASS) # for mvrnorm() and kernel density estimate
library(RColorBrewer) # for color vectors
library(raster) # for creating a raster object
library(pracma) # for computing the gradient
library(mgcv) # for in.out()
library(mvtnorm) # for dnvnorm()
library(matrixStats) # use for row medians
library(truncnorm) # for rtruncnorm()


### -----------------------------------------------------------------
### Generate the preferred habitat and observations
### -----------------------------------------------------------------


### Generate the potential field

potential_field=raster(xmn=-10,xmx=10,ymn=-10,ymx=10,res=0.1) #create a raster template
ncell(potential_field)

for(i in 1:ncell(potential_field)){ # fill the raster template
  print(i)
  xcoord=xFromCell(potential_field,i)
  ycoord=yFromCell(potential_field,i)
  coord=c(xcoord,ycoord)
  xvec <- as.matrix(coord) # get current location
  lvec <- sqrt(sum(xvec^2)) # length of the vector
  fmix=400 # mixing term for the parabola and Gaussian
  potfun <- lvec^2 + fmix*(1-exp(-lvec^2)) # mix of parabola and Gaussian
  potential_field[i]=potfun
}

### Plot the preferred habitat (potential_field)
shadesOfGrey=colorRampPalette(c("grey0", "grey100"))
image(potential_field,col=shadesOfGrey(30))
box()


### Simulate a CTCRW

s=49 # set the seed
set.seed(s)

T=20 # simulation duration
dt=0.05 # time step
nt=T/dt # number of time steps
# note: ensure nt is an integer

time=t=seq(dt,T,dt) # time index vector

# parameters (originals for continuous time models)
sig.w=8 # std dev of the random process per unit time 

# parameters (scaled for use in discrete model, based time step)
sig.w=sig.w*dt # scaled std dev
sig.w

# Parameters for the sine wave used for phi
f=0.4
A=0.2 # amplitude
D=0.8 # vertical shift
phi=matrix(A*cos(f*t)+D,nt,1)

# accumulators for state vector
x=matrix(0,2,nt) # accumulator for position
v=matrix(0,2,nt) # accumulator for velocity 

# initial conditions
x[,1]=c(2,2)
v[,1]=c(-0.5,-0.5)

#accumulator for mu vector (drift term)
mu=matrix(0,2,nt)

# run the loop for recursive generation 
# yields state vector for position (x), and velocity (v)

for (k in 2:nt){
  
  if (k%%10==0)  print(c(s,k))
  
  xcurrent=c(x[1,k-1],x[2,k-1])

  # compute the potential function
  xvec <- as.matrix(xcurrent) # get current location 
  lvec <- sqrt(sum(xvec^2)) # length of the vector
  dvec <- -(1/lvec)*xvec # unit vector pointing to origin
  fmix=400 # how to mix the Gaussian and parabola
  strength <- 2*lvec*(1+fmix*exp(-(lvec^2))) # magnitude of gradient of potential function
  strength <- 0.03*strength # scale the strength
  mu[,k]=strength*dvec # drift term
  
  # Compute the velocity and location at time k
  wv=sig.w*matrix(rnorm(2),2,1) # 2x1 white noise
  v[,k]=(1-phi[k,1])*mu[,k]+phi[k,1]*v[,k-1]+wv # velocity process
  x[,k]=x[,k-1]+dt*v[,k] # retrieve position from velocity    
  
} #end of time loop


# Plot the realization
phi=matrix(A*cos(f*t)+D,nt,1)
redblue=colorRampPalette(c("red","blue")) # blue when phi tends to 1; red when phi tends to 0
phi_col=redblue(10)[as.numeric(cut(as.vector(phi),breaks=10))]
image(potential_field,col=shadesOfGrey(30))
box(which="plot")
points(x[1,],x[2,],col=phi_col,pch=19,cex=0.5,type="b") # initial position
points(x[1,1],x[2,1],col="black",pch=4,cex=2)


### Simulate a data sets based on the realization

# accumulators for observations
y1=matrix(NA,2,nt)

# Pace of available observations
p=1 # 1 means every time step (should be between 0 and 1)

# std of observation noise
sig.y=0.1 # depends on data quality

Nbobs=p[1]*nt # Number of observations available 
time_obs=sort(sample(nt,Nbobs)) # times of observations

for(k in 1:nt){
  if(is.na(match(1:nt,time_obs))[k]==FALSE){ 
    wy=sig.y[1]*matrix(rnorm(2),2,1) # 2x1 white noise
    y1[,k]=x[,k]+wy # observed positions
  }
  else{y1[,k]=c(NA,NA)} # in case of missing data, use NA
}


### -----------------------------------------------------------------
### Forecasting experiment 
### -----------------------------------------------------------------

set.seed(s)

### Function for n-step ahead prediction
nstepahead <- function(xa,v,ns,sig.w,phi) {
  
  wv=sig.w*matrix(rnorm(2),2,1) # 2x1 system noise
  
  # compute potential function
  xvec <- as.matrix(xa) # get current location 
  lvec <- sqrt(sum(xvec^2)) # length of the vector
  dvec <- -(1/lvec)*xvec # unit vector pointing to origin
  fmix=400 # how to mix the Gaussian and parabola
  strength <- 2*lvec*(1+fmix*exp(-(lvec^2))) # magnitude of gradient of potential function
  strength <- 0.03*strength # scale the strength
  mu=strength*dvec
  
  # update velocity and position
  xf <- xvec
  for (i in 1:ns){
    v=(phi*v)+(1-phi)*mu+wv  
    xf=xf+(dt*v)
  }
  return(xf)
}


### Set forecast horizon
nstepvec <- c(1,2,3,4,5,7,9,12,15,20,25,30,40,50)

### Load in the observations (i.e. time, observation)
tobs <- time
yobs <- t(y1)
xtrue <- t(x)
numobs <- length(t)

### Setup

# Time
T=20 # simulation duration
dt=0.05 # time step
time=t=seq(dt,T,dt) # vector of actual time
nt=length(time)

# observation index (if irregular obs)
idx<-match(time,tobs)
idx1<-!is.na(idx)
obsidx<-which(idx1==TRUE)
numobs <- length(obsidx)
yobs <- yobs[1:numobs,]
tobs <- tobs[1:numobs]

# number of variables
nvar=2 # number of variables, 2 since bivariate state for position
nobs=2 # number of obs available at any time step (2 = x & y position)
np=100 # specify number of particles, or sample size

# observation statistics
sig.v=sig.y # observation error std dev
# set up the observation error cov matrix
Id=matrix(0,nobs,nobs)
diag(Id)=1
R=(sig.v^2)*Id # obs error cov

# model statistics
sig.w=8 # process noise std dev

# scale the parameters (for use in discrete model, based on time step)
sig.w=sig.w*dt

# Sine wave parameters
f=0.4
A=0.2 # amplitude
D=0.8 # vertical shift
sdphi=0.05 # standard deviation for phi process (state augmentation)

### Accumulators
# setup initial values for weights (set as equally likely)
w=matrix(1/np,np,1)

# accumulator for phi
phi_acc=matrix(A*cos(f*1)+D,np,1) # accumulator and initial condition for phi (forecast)

# accumulators for state vector (at any given time)
xf_acc=matrix(NA,2,np) # accumulator for position (forecast)
xa_acc=matrix(2,2,np) # initial conditions for position (assimilated)

vf_acc=matrix(NA,2,np) # accumulator for velocity forecast
vf_acc[1,]=rep(-0.5,np) # initial conditions on velocity
vf_acc[2,]=rep(-0.5,np)

va_acc=matrix(NA,2,np) # accumulator for velocity analysis
va_acc[1,]=rep(-0.5,np) # initial conditions on velocity
va_acc[2,]=rep(-0.5,np)

xfn_acc=matrix(NA,2,np) # accumulator for n-step ahead forecast

# Note: initial conditions are set above 

### Ensemble Accumulators: store full sample over time
ensemble=array(0,dim=c(nt,nvar,np)) # assimilation ensemble
ensemble[1,,]=xa_acc # initial ensemble

ensemble_xf=array(NA,dim=c(nt,nvar,np)) # forecast ensemble
ensemble_xfn=array(NA,dim=c(nt,nvar,np,length(nstepvec))) # n-step ahead forecast ensemble
ensemble_vf=array(NA,dim=c(nt,nvar,np)) 
ensemble_vf[1,,]=vf_acc
ensemble_va=array(NA,dim=c(nt,nvar,np)) 
ensemble_phi=matrix(NA,nt,np) # ensemble for phi

# accumulators for posterior mean/medians (use median below)
xbar=matrix(NA,nvar,numobs) # assimilation median
xbar[,1]=rowMedians(xa_acc,na.rm=TRUE) # compute mean/median for initial conditions

xbar_xf=matrix(NA,nvar,numobs) # forecast median
xbar_xf[,1]=rowMedians(xa_acc,na.rm=TRUE) # add mean/median for initial conditions

xbar_xfn=matrix(NA,nvar,numobs) # n-step ahead forecast median
xbar_xfn[,1]=rowMeans(xa_acc,na.rm=TRUE) # add mean/median for initial conditions   

xbar_vf=matrix(NA,nvar,numobs) # n-step ahead forecast median
xbar_vf[,1]=rowMeans(va_acc,na.rm=TRUE) # add mean/median for initial conditions

xbar_va=matrix(NA,nvar,numobs) # n-step ahead forecast median

phi_bar=matrix(NA,1,numobs)
phi_bar[1]=median(phi_acc,na.rm=TRUE) # initial values for phi

rmsep=matrix(NA,nt,length(nstepvec)) # vector for rmse of persistence forecast
rmsef=matrix(NA,nt,length(nstepvec)) # vector for rmse of forecast

# monitoring
neff=array(NA, dim=c(nt,1)) # effective number of particles


####################################################
# DATA ASSIMIlATION: CTCRW MODEL AND PARTICLE FILTER
####################################################

for (k in 2:nt){    # loop over time
  
  if (k%%10==0) {write(k,"")} # console output every 10th time step
  
  # ===============
  # prediction step - yields forecast/predictive ensemble
  # ===============
  
  for (ip in 1:np){
    
    xa <- xa_acc[,ip] 
    xa <- as.matrix(xa)
    v <- va_acc[,ip]
    
    # compute process white noise
    wv=sig.w*matrix(rnorm(2),2,1) # 2x1 system noise
    
    # estimate phi (state augmentation)
    phiold <- phi_acc[ip] # initial condition 
    phi=rtruncnorm(1, a=0, b=1, mean = phiold, sd = sdphi) #constrains between 0 and 1
    phi_acc[ip]=phi # store result for phi
    
    # compute the potential function
    xvec <- as.matrix(xa) # get current location 
    lvec <- sqrt(sum(xvec^2)) # length of the vector
    dvec <- -(1/lvec)*xvec # unit vector pointing to origin
    fmix=400 # how to mix the Gaussian and parabola
    strength <- 2*lvec*(1+fmix*exp(-(lvec^2))) # magnitude of gradient of potential function
    strength <- 0.03*strength # scale the strength
    mu=strength*dvec
    
    # one step ahead forecast for use in assimilation step
    v=(phi*v)+(1-phi)*mu+wv # update velocity
    xf=xa+(dt*v) # update position
    
    xf_acc[,ip]=xf # store forecast positions
    vf_acc[,ip]=v # store forecast velocity
    
  }  # end ip
  
  xbar_xf[,k]=rowMedians(xf_acc,na.rm=TRUE) # compute mean/median  
  xbar_vf[,k]=rowMedians(vf_acc,na.rm=TRUE) # compute mean/median 
  
  # ===============
  # assimilation step
  # ===============
  
  # below, use if statement to check if there is an observation at the current time
  # if so, assimilate (do particle filter observation update to get posterior sample)
  # if not, use forecast ensemble as posterior sample
  
  y=matrix(yobs[k,],2,1)
  
  if(is.na(y[1,])|is.na(y[2,])==TRUE){
    
    xa_acc <- xf_acc
    va_acc <- vf_acc
    phi_acc <- phi_acc
    
  }
  
  else{
    # assign weights to each particle (proportional to likelihood)
    for (ip in 1:np){
      w[ip] <- dmvnorm(xf_acc[,ip], mean=y, sigma=R)
    }
    w=(1/(sum(w)))*w # normalize the weights
    
    # carry out weighted resampling
    m=sample(1:np,size=np,prob=w,replace=T) # index for particle to be resampled
    xa_acc=xf_acc[,m] # choose index particles from forecast ensemble
    va_acc=vf_acc[,m] # choose index particles from forecast ensemble
    phi_acc=phi_acc[m] # resample particle ensemble
    neff[k]=length(unique(m))
    
  }
  
  xbar[,k]=rowMeans(xa_acc,na.rm=TRUE) # compute median
  xbar_va[,k]=rowMeans(va_acc,na.rm=TRUE) # compute median
  phi_bar[k]=median(phi_acc,na.rm=TRUE) # compute mean/median
  
  # ===============
  #  n-step-ahead forecasts and store
  # ===============
  
  # loop over all forecast horizons
  for (i in 1:length(nstepvec)){
    for (ip in 1:np){
      
      xic <- rowMedians(xa_acc,na.rm=TRUE)
      vic <- rowMedians(va_acc,na.rm=TRUE)
      phic <- median(phi_bar[k],na.rm=TRUE)
      nstep <- nstepvec[i]
      
      xfn <- nstepahead(xic,vic,nstep,sig.w,phic) # forecast (call function)
      xfn_acc[,ip]=xfn 
      
    } # end of ip loop
    
    xbar_xfn[,k]=rowMedians(xfn_acc,na.rm=TRUE)  # compute mean/median
    ensemble_xfn[k,,,i]=xfn_acc # n-step ahead forecast particle ensemble

  } # end of i loop 
  
  # ===============
  # store the results
  # ===============
  
  ensemble[k,,]=xa_acc # full particle ensemble
  ensemble_xf[k,,]=xf_acc # forecast particle ensemble
  ensemble_vf[k,,]=vf_acc # n-step ahead forecast particle ensemble
  ensemble_va[k,,]=va_acc # n-step ahead forecast particle ensemble
  ensemble_phi[k,]=t(phi_acc) # forecast particle ensemble
  
} # end of k loop for time steps



### ------------------------------------------------------------
### Visualization
### ------------------------------------------------------------

# TRACK
image(potential_field,col=shadesOfGrey(30),xlab="x",ylab="y")
box()
matplot(2*ensemble[,1,], 2*ensemble[,2,], type="p", col="lightblue", pch=19, cex=0.05,add=TRUE)
lines(2*x[1,],2*x[2,],col="black",lwd=1) #the initial CTCRW
points(2*yobs[,1],2*yobs[,2],pch=19, col="red", cex=0.3)
points(2*xbar[1,],2*xbar[2,],col="blue",pch=19, cex=0.2)
points(2*xbar[1,1],2*xbar[2,1],pch=4,cex=2)
# Note: positions are non-dimensional (2*x)

# PHI ESTIMATE
tnd=2*time*sig.w # Non-dimensional time vector
matplot(tnd,ensemble_phi[,], type="p", pch=19, cex=0.05, col="lightgray", 
        ylab="", xlab="Time", lty=1, main="")
mtext(expression(phi[t]),side=2,line=2.5,cex=0.8)
cosine_wave=matrix(A*cos(f*t)+D,nt,1)
lines(tnd,cosine_wave,pch=16,lwd=0.7)
points(tnd,phi_bar, pch=19, cex=0.2, col="blue",type="b")
lo=smooth.spline(tnd, phi_bar[1,], spar=1)
lines(predict(lo),col="dodgerblue",lwd=1.5)


### ------------------------------------------------------------
### Compute the Root Mean Square Error (RMSE)
### ------------------------------------------------------------

for(i in 1:length(nstepvec)){
  
  nstep=nstepvec[i]
  
  for(k in 1:(nt-nstep)){
    
    ##### Metrics relative to observations
    
    xbar_xfn_x_y=cbind(matrix(ensemble_xfn[k,1,,i]),matrix(ensemble_xfn[k,2,,i]))
    xbar_xfn_x_y=colMedians(xbar_xfn_x_y,na.rm=TRUE) #median of ensemble
    
    # Forecast 
    res_forecast <- yobs[k+nstep,] - xbar_xfn_x_y
    rmsef[k,i] <- sqrt(mean(res_forecast^2,na.rm=TRUE))
    
    # Persistence forecast
    res_persist <- yobs[k+nstep,] - t(xbar[,k])
    rmsep[k,i] <- sqrt(mean(res_persist^2,na.rm=TRUE))
    
  } # end of k loop
} # end of i loop


# PLOT RMSE AGAINST TIME
tnd=nstepvec*2*sig.w # non-dimensional time vector
plot(tnd,log(colMedians(rmsef,na.rm=TRUE)+1),type="l",lty=2,lwd=1.1,col="blue"
     ,xlab="Forecast horizon",ylab=expression(log(RMSE + 1)))
lines(tnd,log(colMedians(rmsep,na.rm=TRUE)+1),type="l",lty=1,lwd=1.1,col="black")
legend("topleft",legend=c("Forecast","Persist. forecast"),lwd=rep(1.1,2),lty=c(2,1),col=c("blue","black"))

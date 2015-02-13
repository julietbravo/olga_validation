import numpy as np
from pylab import *
import glob
from copy import deepcopy
from scipy.ndimage.filters import gaussian_filter
from mpl_toolkits.basemap import Basemap

close('all')

R = 6371.   # radius earth

def key_nearest(array,value):
    """ Find key of nearest element (value) in list (array) """

    return (np.abs(array-value)).argmin()

def value_nearest(array,value):
    """ Find nearest value (value) in list (array) """

    return array[(np.abs(array-value)).argmin()]

def distance1(lon1, lat1, lon2, lat2):
    """ Distance (haversine, in km) between coordinates) """

    lon1,lat1,lon2,lat2 = map(radians,[lon1,lat1,lon2,lat2])
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    return R * c

def angle1(lon1,lat1,lon2,lat2):
    """ Angle between two coordinates """

    lon1,lat1,lon2,lat2 = map(radians,[lon1,lat1,lon2,lat2])
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1)*np.sin(lat2)-np.sin(lat1)*np.cos(lat2)*np.cos(dlon)
    a = np.arctan2(y,x) * 180./np.pi

def distance2(lon1,lat1,lon2,lat2):
    """ Simplified version of distance between two coordinates """

    lon1,lat1,lon2,lat2 = map(radians,[lon1,lat1,lon2,lat2])
    x = (lon2-lon1)*np.cos((lat1+lat2)/2.)
    y = lat2-lat1
    return sqrt(x*x + y*y) * 6371. 

def angle2(lon1,lat1,lon2,lat2):
    """ Simplified version of angle between coordinates """

    lon1,lat1,lon2,lat2 = map(radians,[lon1,lat1,lon2,lat2])
    a = np.arctan2(lon2-lon1,lat2-lat1) * 180./np.pi 
    return (a+360)%360

def point(lon1,lat1,d,angle):
    """ Coordinate given start, distance and direction """

    angle = angle*np.pi/180.
    lon1,lat1 = map(radians,[lon1,lat1])
    lat2 = np.arcsin(np.sin(lat1)*np.cos(d/R)+np.cos(lat1)*np.sin(d/R)*np.cos(angle))
    lon2 = lon1 + np.arctan2(np.sin(angle)*np.sin(d/R)*np.cos(lat1),np.cos(d/R)-np.sin(lat1)*np.sin(lat2))
    return (lon2*180./np.pi,lat2*180./np.pi)

def spectral_filter(var,k):
    """ Spectral filter, k=wavenumber """

    var_c = deepcopy(var)
    fft = np.fft.rfft(var_c)
    fft[k:] = 0.
    var_c = np.fft.irfft(fft)
    return var_c

"""
Global switch between simple and more precise angle and distance calculations:
"""
distance = distance2
angle    = angle2

class readigc:
    def __init__(self,igcfile):
        """ Class to read and analyse IGC files. Args: igcfile, full file path to IGC file """

        print('reading %s'%igcfile)
        self.igcfile = igcfile
        
        # Loop through IGC file to find number of B-records
        # Probably faster than working with lists and appending (?)
        self.pol_file()

        # Create empty arrays
        self.t   = np.zeros(self.n_records) # time of record
        self.lat = np.zeros(self.n_records) # latitude
        self.lon = np.zeros(self.n_records) # longitude
        self.zp  = np.zeros(self.n_records) # altitude pressure sensor
        self.zg  = np.zeros(self.n_records) # altitude GPS

        # Read the IGC file & analyse it
        self.read_file()
        self.analyse_file()

    def pol_file(self):
        """ Find the number of B-records """

        b = 0
        f = open(self.igcfile,'r')
        for line in f:
            if(line[0] == 'B'):
                b += 1 
        self.n_records = b-1 if b%2 != 0 else b

    def read_file(self):
        """ Read IGC file into arrays """

        f = open(self.igcfile,'r')
        l = 0
        for line in f:
            if(line[0] == 'B' and l<self.n_records):
                self.t[l]   = float(line[1:3]) + float(line[3:5]) / 60. + float(line[5:7]) / 3600.
                sign_lat    = +1 if(line[14] == "N") else -1
                sign_lon    = +1 if(line[23] == "E") else -1
                self.lat[l] = sign_lat * float(line[7:9])   + float(line[9:11])  / 60. + float(line[11:14]) / 60000.
                self.lon[l] = sign_lon * float(line[15:18]) + float(line[18:20]) / 60. + float(line[20:23]) / 60000.
                self.zp[l]  = float(line[25:30])
                self.zg[l]  = float(line[30:35])
                l += 1

    def analyse_file(self):
        """ Analyze the IGC file: find updrafts, etc. """

        z = self.zp  # switch for which altitude to use     
        dt_glob = (self.t[1] - self.t[0]) * 3600. # global dt

        self.th = np.zeros(self.n_records-1)  # time in between fixes
        self.b = np.zeros(self.n_records-1) # bearing from fix to fix
        self.w = np.zeros(self.n_records-1) # vertical velocity from fix to fix
        self.dbdt = np.zeros(self.n_records) # (abs) change in bearing 

        for i in range(self.n_records-1):
            dt = (self.t[i+1] - self.t[i]) * 3600.
            self.th[i] = 0.5 * (self.t[i] + self.t[i+1])
            self.b[i]  = angle(self.lon[i],self.lat[i],self.lon[i+1],self.lat[i+1])
            self.w[i]  = (z[i+1] - z[i]) / dt

        for i in range(1,self.n_records-1):
            dt = (self.th[i] - self.th[i-1]) * 3600.
            b1 = np.abs(self.b[i] - self.b[i-1])
            b2 = np.abs((self.b[i]+360) - self.b[i-1])
            b3 = np.abs(self.b[i] - (self.b[i-1]+360))
            self.dbdt[i] = min([b1,b2,b3]) / dt

        dbdt_lim = 5. # minimal change in bearing [deg/sec] to count as updraft
        min_time_thermal = 60. # minimal time spent in thermal [sec] to count as updraft
        min_dz_thermal = 50. # minimal height gain to count as updraft
        min_w_thermal = 0.1 # minimal average velocity to count as updraft

        cutoff_wl = 60. # Wave length [sec] on which to filter 
        wn = ((self.t[-1] - self.t[0]) * 3600.) / cutoff_wl # cut-off wavenumber
        self.dbdtf = spectral_filter(self.dbdt,wn) # Spectral filter to remove small fluctuations

        self.t_thermal = []
        self.t0_thermal = []
        self.t1_thermal = []
        self.lat_thermal = []
        self.lon_thermal = []
        self.w_thermal = []
        self.dz_thermal = []
        i = 0
        while(i < self.n_records-1):
            if(self.dbdtf[i] > dbdt_lim):
                n = 0
                while(self.dbdtf[i+n] > dbdt_lim and i+n < self.n_records-1):
                    n += 1 
                if(n * dt_glob > min_time_thermal and \
                    (z[i+n] - z[i]) > min_dz_thermal and \
                    np.mean(self.w[i:i+n]) > min_w_thermal):
 
                    self.t0_thermal.append(i)
                    self.t1_thermal.append(i+n)
                    self.t_thermal.append(np.mean(self.t[i:i+n]))
                    self.lat_thermal.append(np.mean(self.lat[i:i+n]))
                    self.lon_thermal.append(np.mean(self.lon[i:i+n]))
                    self.w_thermal.append(np.mean(self.w[i:i+n]))
                    self.dz_thermal.append(z[i+n] - z[i])
                i += max([1,n])
            else:
                i += 1

    def quickplot(self):
        """ Very simple plot for validation """

        figure()
        subplot(141)
        scatter(self.lon, self.lat)
        xlabel('lon')
        ylabel('lat')

        subplot(142)
        plot(self.t, self.zp, label='pressure alt')
        plot(self.t, self.zg, label='GPS alt')
        xlabel('time [h]')
        ylabel('z [m]')
        legend()     

        subplot(143)
        plot(self.th, self.w)
        xlabel('time [h]')
        ylabel('w [m/s]')

        subplot(144)
        plot(self.t, self.dbdt, label='raw')
        plot(self.t, self.dbdtf, label='filtered')
        xlabel('time [h]')
        ylabel('dbdt [deg/s]')
        legend()     

if __name__ == "__main__":
    """ Read all IGC files from specified directory """

    files = glob.glob('reference_data/igc/20140702/*')
    print('Found %i IGC files'%np.size(files))

    igcs = []
    for igc in files[:]:
        igcobj = readigc(igc)
        igcs.append(igcobj)
    
    #test = readigc(files[0])
    #test.quickplot()


    if(True):
        """ Sample flights based on time and spatial location """

        for t0 in range(7,20,2):
            t1 = t0+2
            dlon = 1
            dlat = 0.5
            cent_lon = np.arange(0,15.001,dlon)
            cent_lat = np.arange(46,54,dlat)
    
            w_grid = np.zeros((cent_lon.size,cent_lat.size))
            n_samp = np.zeros_like(w_grid)
    
            for igc in igcs:
                for t in range(np.size(igc.t_thermal)):
                    if(igc.t_thermal[t] >= t0 and igc.t_thermal[t] <= t1):
                        i = key_nearest(cent_lon,igc.lon_thermal[t])
                        j = key_nearest(cent_lat,igc.lat_thermal[t])
                        w_grid[i,j] += igc.w_thermal[t]
                        n_samp[i,j] += 1
                        
            w_grid[np.where(n_samp>0)] /= n_samp[np.where(n_samp>0)]
            w_grid = np.ma.masked_where(n_samp == 0,w_grid)
    
            fig   = figure(figsize=(10.5,7.0))
            axloc = [0.03,0.05,0.85,0.87]  # left,bottom,width,height
            ax    = fig.add_axes(axloc)
            ax.set_title('Updraft velocity between t=%6.3f and t=%6.3f UTC'%(t0,t1),loc='left')
            
            m = Basemap(width=700000,height=500000,
                        rsphere=(6378137.00,6356752.3142),\
                        resolution='i',area_thresh=10.,projection='lcc',\
                        lat_1=51,lat_0=51,lon_0=8)
            
            m.drawcoastlines(linewidth=1,color='0.5')
            m.drawcountries(linewidth=1,color='0.5')
            m.drawrivers(linewidth=0.5,color='#0066FF')
            m.drawmapboundary()
    
            lons2 = cent_lon - 0.5 * dlon
            lats2 = cent_lat - 0.5 * dlat
    
            lons,lats = np.meshgrid(lons2,lats2)
            lon,lat = m(lons,lats)
            cmap = cm.get_cmap('jet', 8)
            cf = m.pcolormesh(lon,lat,np.transpose(w_grid),cmap=cmap,vmin=0,vmax=4)
    
            for igc in igcs:
                lon,lat=m(igc.lon,igc.lat)
                m.plot(lon,lat,color='k',alpha=0.4,linewidth=1)
                lon,lat=m(igc.lon_thermal,igc.lat_thermal)
                #cf = m.scatter(lon,lat,s=30,c=igc.w_thermal,vmin=0,vmax=4,edgecolor='none',cmap=cmap)
    
            pos = ax.get_position()
            l,b,w,h = pos.bounds
            cax = axes([l+w,b+0.1,0.01,h-0.2])
            cb=colorbar(cf,drawedges=False,cax=cax,ticks=[0,0.5,1,1.5,2,2.5,3,3.5,4])
            cb.ax.tick_params(labelsize=8) 
            cb.outline.set_color('white')
            cb.set_label('Updraft velocity [m/s]')
            
            savefig('w_%02i_%02i_02072014.png'%(t0,t1))
    
    if(False):
        """ Plot map with all flights """

        fig   = figure(figsize=(10.5,8.0))
        axloc = [0.03,0.05,0.85,0.87]  # left,bottom,width,height
        ax    = fig.add_axes(axloc)
        ax.set_title('OLC+Skylines, 02-07-2014',loc='left')
        
        m = Basemap(width=700000,height=600000,
                    rsphere=(6378137.00,6356752.3142),\
                    resolution='i',area_thresh=10.,projection='lcc',\
                    lat_1=51,lat_0=51,lon_0=8)
        
        m.drawcoastlines(linewidth=1,color='0.5')
        m.drawcountries(linewidth=1,color='0.5')
        m.drawrivers(linewidth=0.5,color='#0066FF')
        m.drawmapboundary()
        
        cmap = cm.get_cmap('jet', 8)
        
        for igc in igcs:
            lon,lat=m(igc.lon,igc.lat)
            m.plot(lon,lat,color='k',alpha=0.4,linewidth=1)
            lon,lat=m(igc.lon_thermal,igc.lat_thermal)
            cf = m.scatter(lon,lat,s=30,c=igc.w_thermal,vmin=0,vmax=4,edgecolor='none',cmap=cmap)
        
        pos = ax.get_position()
        l,b,w,h = pos.bounds
        cax = axes([l+w,b+0.1,0.01,h-0.2])
        cb=colorbar(cf,drawedges=False,cax=cax)
        cb.ax.tick_params(labelsize=8) 
        cb.outline.set_color('white')
        cb.set_label('Updraft velocity [m/s]')
        
        savefig('flights_02072014.png')
    


import numpy as np

def readblock(txtobj, column, t0, t1):
    values = txtobj[t0:t1,column]
    try:
        values = np.array(values, dtype=np.float32)
    except:
        for i in range(np.size(values)):
            if(values[i].strip(' ') == ''):
                values[i] = -1         
        values = np.array(values, dtype=np.float32)

    # Mask missing values
    values = np.ma.masked_where(values==-1, values)

    return values

class readfile:
    def __init__(self, filename, year, month, day):
        print('Reading %s'%filename)

        txt     = np.genfromtxt(filename, dtype='str', delimiter=',', skip_header=32)
        dstring = txt[:,1]
        t0 = np.where(dstring == '%04i%02i%02i'%(year, month, day))[0][0]
        t1 = np.where(dstring == '%04i%02i%02i'%(year, month, day))[0][-1]+1

        i        = 2
        var      = []
        varname  = []

        self.HH  = readblock(txt, i, t0, t1) ; i+= 1 
        self.DD  = readblock(txt, i, t0, t1) ; i+= 1 
        self.FH  = readblock(txt, i, t0, t1) ; i+= 1 
        self.FF  = readblock(txt, i, t0, t1) ; i+= 1 
        self.FX  = readblock(txt, i, t0, t1) ; i+= 1 
        self.T   = readblock(txt, i, t0, t1) ; i+= 1 
        self.T10 = readblock(txt, i, t0, t1) ; i+= 1 
        self.TD  = readblock(txt, i, t0, t1) ; i+= 1 
        self.SQ  = readblock(txt, i, t0, t1) ; i+= 1 
        self.Q   = readblock(txt, i, t0, t1) ; i+= 1 
        self.DR  = readblock(txt, i, t0, t1) ; i+= 1 
        self.RH  = readblock(txt, i, t0, t1) ; i+= 1 
        self.P   = readblock(txt, i, t0, t1) ; i+= 1 
        self.VV  = readblock(txt, i, t0, t1) ; i+= 1 
        self.N   = readblock(txt, i, t0, t1) ; i+= 1 
        self.U   = readblock(txt, i, t0, t1) ; i+= 1 
        self.WW  = readblock(txt, i, t0, t1) ; i+= 1 
        self.IX  = readblock(txt, i, t0, t1) ; i+= 1 
        self.M   = readblock(txt, i, t0, t1) ; i+= 1 
        self.R   = readblock(txt, i, t0, t1) ; i+= 1 
        self.S   = readblock(txt, i, t0, t1) ; i+= 1 
        self.O   = readblock(txt, i, t0, t1) ; i+= 1 
        self.Y   = readblock(txt, i, t0, t1) ; i+= 1                       

        # Convert / cleanup
        # Some low values are flagged as -1, set to zero
        self.RH[self.RH == -1] = 0
        self.SQ[self.SQ == -1] = 0

        # conversions from integer to float:
        self.FH  /= 10. 
        self.FX  /= 10.
        self.T   /= 10.
        self.T10 /= 10.
        self.TD  /= 10.
        self.SQ  /= 10.
        self.DR  /= 10.
        self.RH  /= 10.
        self.P   /= 10.
        self.N   /= 8.  # from octa to fraction
  
# Simple test if script directly called       
if __name__ == "__main__":
    from pylab import *

    stations=([210,215,225,235,240,242,249,251,257,260,265,267,269,270,273,275,277,278,\
               279,280,283,286,290,310,319,323,330,340,344,348,350,356,370,375,377,380,391])
    figure()
    for s in stations:  
        try: 
            data = readfile('reference_data/knmi/uurgeg_%i_2011-2020.txt'%s, 2011, 1, 6)        
            plot(data.HH, data.RH)
        except:
            pass
 

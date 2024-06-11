import numpy as np
from astropy.constants import G,c
import astropy.units as u
from scipy.optimize import root
from scipy.integrate import quad

def orbital_elements_to_vectors(a, e, cos_i, Omega, omega, f, m=1, units=(u.AU,u.km/u.s,u.Msun)):
    ''' 
    Input:
        a: semi-major axis (AU)
        e: eccentricity
        cos_i: Cos of inclination
        Omega: longitude of the ascending node
        omega: argument of periapsis
        M: true anomaly
        m: total mass (Msun)

    Output:
        rvec: relative position vector (AU)
        vvec: relative velocity vector (km/s)
    '''

    # Raise error if input is not a number
    if(not all(isinstance(x,(int,float)) for x in [a,e,cos_i,Omega,omega,f,m])):
        raise ValueError('All input parameters must be numbers')

    m *= G.to(units[0]*units[1]**2/units[2]).value

    p = (1-e**2)*a
    r = p/(1+e*np.cos(f))
    u1 = np.array([np.cos(Omega),
                   np.sin(Omega),
                   0])
    u2 = np.array([-cos_i*np.sin(Omega),
                   cos_i*np.cos(Omega),
                   np.sqrt(1-cos_i**2)])
    rvec = r*(u1*np.cos(f+omega)+u2*np.sin(f+omega))
    vvec = np.sqrt(m/p)*(-u1*(e*np.sin(omega)+np.sin(f+omega))+u2*(e*np.cos(omega)+np.cos(f+omega)))

    return rvec,vvec

def orbital_elements_to_vectorial_elements(cos_i, Omega, omega):
    ''' 
    Input:
        a: semi-major axis (AU)
        e: eccentricity
        cos_i: Cos of inclination
        Omega: longitude of the ascending node
        omega: argument of periapsis
        m: total mass (Msun)

    Output:
        lvec: unit vector in the direction of the angular momentum
        evec: unit vector in the direction of the eccentricity vector
        nvec: unit vector in the direction of the line of nodes
    '''

    # Raise error if input is not a number
    if(not all(isinstance(x,(int,float)) for x in [cos_i,Omega,omega])):
        raise ValueError('All input parameters must be numbers')
    
    # Raise error if cos_i is not between -1 and 1
    if(cos_i<-1 or cos_i>1):
        raise ValueError('Cosine of inclination must be between -1 and 1')
    
    sin_i = np.sqrt(1-cos_i**2)

    # From Merritt's book, E.q. (4.59)

    lvec = np.array([sin_i*np.sin(Omega),
                     -sin_i*np.cos(Omega),
                     cos_i])
    
    evec = np.array([np.cos(omega)*np.cos(Omega)-np.sin(omega)*np.sin(Omega)*cos_i,
                     np.cos(omega)*np.sin(Omega)+np.sin(omega)*np.cos(Omega)*cos_i,
                     np.sin(omega)*sin_i])
    
    nvec = np.array([-np.sin(omega)*np.cos(Omega)-np.cos(omega)*np.sin(Omega)*cos_i,
                     -np.sin(omega)*np.sin(Omega)+np.cos(omega)*np.cos(Omega)*cos_i,
                     np.cos(omega)*sin_i])

    return lvec,evec,nvec

def orbital_angular_momentum(a, e, m1=1, m2=1, units=(u.AU,u.km/u.s,u.Msun)):
    '''
    Input:
        a: semi-major axis (AU)
        e: eccentricity
        m1: mass of the primary (Msun)
        m2: mass of the secondary (Msun)
        units: units of the input/output parameters

    Output:
        L: orbital angular momentum
    '''

    # Raise error if input is not a number
    if(not all(isinstance(x,(int,float)) for x in [a,e,m1,m2])):
        raise ValueError('a,e,m1,m2 must be numbers')
    
    # Raise error if a is negative
    if(a<0):
        raise ValueError('a must be positive')
    
    # Raise error if m1, m2 are negative or if e is not between 0 and 1
    if(m1<0 or m2<0 or e<0 or e>=1):
        raise ValueError('m1,m2 must be positive and e must be between 0 and 1')
    
    m1 *= G.to(units[0]*units[1]**2/units[2]).value
    m2 *= G.to(units[0]*units[1]**2/units[2]).value

    m = m1+m2
    mu = m1*m2/m

    L = mu*np.sqrt(m*a*(1-e**2))

    return L
'''
def merger_time(a, e, m1=1, m2=1, F='Numerical_Integration', units=(u.AU,u.yr,u.Msun)):
    
    a *= units[0]
    m1 *= units[2]
    m2 *= units[2]

    m = m1+m2
    mu = m1*m2/m

    g = lambda e : e**(12/19)/(1-e**2)*(1+(121/304)*e**2)**(870/2299)

    if e==0:
        t = 5/256*(c**5/G**3)*a**4/mu/m**2
        return t.to(units[1]).value
    
    elif F=='Numerical_Integration':
        F0 = 48/19/g(e)**4*quad(lambda e : g(e)**4*(1-e**2)**(5/2)/e/(1+121/304*e**2),0,e)[0]

    elif F=='Low_eccentricity':
        F0 = e**(48/19)/g(e)**4

    elif F=='High_eccentricity':
        F0 = 768/429*(1-e**2)**(7/2)

    else:
        raise ValueError('F must be Numerical_Integration, Low_eccentricity or High_eccentricity')
    
    T0 = orbital_period(a,m=m,units=units)

    return t.to(units[1]).value
'''
def vectors_to_orbital_elements(rvec, vvec, m=1, units=(u.AU,u.km/u.s,u.Msun)):
    ''' 
    Input:
        rvec: relative position vector (AU)
        vvec: relative velocity vector (km/s)
        m: total mass (Msun)

    Output:
        a: semi-major axis (AU)
        e: eccentricity
        cos_i: Cos of inclination
        Omega: longitude of the ascending node
        omega: argument of periapsis
        f: true anomaly
    '''

    # Raise error if vectors are not 3D
    if(len(rvec)!=3 or len(vvec)!=3):
        raise ValueError('Input vectors must be 3D')
    
    # Raise error if input is not a numpy array containing numbers or mass is not a number
    if(not isinstance(rvec,np.ndarray) or not isinstance(vvec,np.ndarray) or not isinstance(m,(int,float))):
        raise ValueError('Input vectors must be numpy arrays and mass must be a number')

    m *= G.to(units[0]*units[1]**2/units[2]).value

    r = np.linalg.norm(rvec)
    v = np.linalg.norm(vvec)
    h = np.cross(rvec,vvec)
    H = np.linalg.norm(h)
    n = np.cross([0,0,1],h)
    N = np.linalg.norm(n)

    evec = np.cross(vvec,h)/m - rvec/r

    e = np.linalg.norm(evec)

    a = 1/(2/r-v**2/m)

    cos_i = h[2]/H

    Omega = np.arccos(n[0]/N)
    if(n[1]<0): Omega = 2*np.pi-Omega

    f = np.arccos(np.dot(rvec,evec)/(r*e))
    if(np.dot(rvec,vvec)<0): f = 2*np.pi-f

    omega = np.arccos(np.dot(n,evec)/(N*e))
    if(evec[2]<0): omega = 2*np.pi-omega

    return a,e,cos_i,Omega,omega,f

def get_true_anomaly(e,M=None):
    '''
    Input:
        e: eccentricity
        M: mean anomaly (if not proveded, a random value is chosen between 0 and 2*pi)

    Output:
        f: true anomaly
    '''

    # Raise error if input is not a number
    if(not isinstance(e,(int,float))):
        raise ValueError('Eccentricity must be a number')
    
    # Raise error if e is not between 0 and 1
    if(e<0 or e>=1):
        raise ValueError('Eccentricity must be between 0 and 1')
    
    # Raise error if M is not None or a number
    if(M is not None and not isinstance(M,(int,float))):
        raise ValueError('Mean anomaly must be None or a number')

    # Calculate true anomaly from random mean anomaly
    if M is None:
        M = np.random.uniform(0,2*np.pi)
    sol = root(lambda E : E-e*np.sin(E)-M, np.random.uniform(0,2*np.pi))
    E = sol.x[0]
    beta = e/(1+np.sqrt(1-e**2))
    f = E+2*np.arctan(beta*np.sin(E)/(1-beta*np.cos(E)))

    return f

def apply_kick_to_orbit(a, vkick, m_SN, dm_SN, m_comp, dm_comp=0, vkick_phi=None, vkick_theta=None, e=0, cos_i=1, Omega=0, omega=0, f=None, v_com=np.zeros(3), units=(u.AU,u.km/u.s,u.Msun), verbose=False):
    '''
    Input:
        a: semi-major axis (AU)
        vkick: kick velocity (km/s)
        m_SN: mass of the exploding star before supernova (Msun)
        dm_SN: mass loss of the supernova (Msun)
        m_comp: mass of the companion (Msun)
        dm_comp: mass loss of the companion, e.g., through winds (Msun)
        vkick_phi: azimuthal angle of the kick velocity (rad)
        vkick_theta: polar angle of the kick velocity (rad)
        e: eccentricity
        cos_i: Cos of inclination
        Omega: longitude of the ascending node
        omega: argument of periapsis
        f: true anomaly
        v_com: centre of mass velocity (km/s)
        units: units of the input parameters
        verbose: print output

    Output:
        a_new: semi-major axis (AU)
        e_new: eccentricity
        cos_i_new: Cos of inclination
        Omega_new: longitude of the ascending node
        omega_new: argument of periapsis
        f_new: true anomaly
        v_com_new: centre of mass velocity (km/s)   
    '''

    # Raise error if input is not a number or None
    if(not all(isinstance(x,(int,float)) or x is None for x in [f,vkick_phi,vkick_theta])):
        raise ValueError('f,vkick_phi,vkick_theta must be numbers or None')
    
    # Raise error if input is not a number
    if(not all(isinstance(x,(int,float)) for x in [a,vkick,m_SN,dm_SN,m_comp,dm_comp])):
        raise ValueError('a,vkick,m_SN,dm_SN,m_comp,dm_comp must be numbers')
    
    # Raise error if verbose is not a boolean   
    if(not isinstance(verbose,bool)):
        raise ValueError('verbose must be a boolean')
    
    # Raise error if v_com is not a numpy array containing numbers
    if(not isinstance(v_com,np.ndarray) or not all(isinstance(x,(int,float)) for x in v_com)):
        raise ValueError('v_com must be a numpy array containing numbers')
    
    # Raise error if a, vkick, m_SN, dm_SN, m_comp, dm_comp are negative or if e is not between 0 and 1
    if(a<0 or vkick<0 or m_SN<0 or dm_SN<0 or m_comp<0 or dm_comp<0 or e<0 or e>=1):
        raise ValueError('a,vkick,m_SN,dm_SN,m_comp,dm_comp must be positive and e must be between 0 and 1')
    
    # Raise error if dm_SN, dm_comp are greater than m_SN, m_comp
    if(dm_SN>m_SN or dm_comp>m_comp):
        raise ValueError('dm_SN,dm_comp must be less than m_SN,m_comp')

    if f is None:
        f = get_true_anomaly(e)

        if verbose:
            print('True anomaly not provided. Calculated from random mean anomaly:',M,end='\n\n')

    if vkick_phi is None:
        vkick_phi = np.random.uniform(0,2*np.pi)

        if verbose:
            print('Azimuthal angle of the kick velocity not provided. Randomly chosen:',vkick_phi,end='\n\n')

    if vkick_theta is None:
        vkick_theta = np.arccos(np.random.uniform(-1,1))

        if verbose:
            print('Polar angle of the kick velocity not provided. Randomly chosen:',vkick_theta,end='\n\n')

    # Orbital vectors before supernova
    rvec_old,vvec_old = orbital_elements_to_vectors(a, e, cos_i, Omega, omega, f, m=m_SN+m_comp, units=units)

    # Velocity vectors of each star before supernova
    vvec1_old = vvec_old*m_comp/(m_SN+m_comp) + v_com
    vvec2_old = -vvec_old*m_SN/(m_SN+m_comp) + v_com

    # Apply kick to component 1
    vkick_vec = np.array([vkick*np.sin(vkick_theta)*np.cos(vkick_phi),
                          vkick*np.sin(vkick_theta)*np.sin(vkick_phi),
                          vkick*np.cos(vkick_theta)])
    vvec1_new = vvec1_old + vkick_vec 

    # Calculate new relative velocity
    vvec_new = vvec1_new - vvec2_old

    # Calculate new centre of mass velocity
    v_com_new = ((m_SN-dm_SN)*vvec1_new + (m_comp-dm_comp)*vvec2_old)/(m_SN+m_comp-dm_SN-dm_comp)

    # Calculate new orbital elements
    a_new,e_new,cos_i_new,Omega_new,omega_new,f_new = vectors_to_orbital_elements(rvec_old, vvec_new, m=m_SN+m_comp-dm_SN-dm_comp, units=units)

    # Print output
    if verbose:
        print('Old masses:',m_SN,m_comp)
        print('Old semi-major axis:',a)
        print('Old eccentricity:',e)
        print('Old inclination:',np.arccos(cos_i))
        print('Old relative velocity:',np.linalg.norm(vvec_old))
        print('Old com velocity:',np.linalg.norm(v_com),end='\n\n')

        print('Kick velocity:',vkick,end='\n\n')

        # Print if orbit is unbound
        if e_new >= 1 or e_new < 0 or a_new <= 0 or ~np.isfinite(a_new) or ~np.isfinite(e_new) or ~np.isfinite(cos_i_new) or ~np.isfinite(Omega_new) or ~np.isfinite(omega_new) or ~np.isfinite(f_new) or ~np.isfinite(v_com_new).all():
            print('Orbit gets unbound.',end='\n\n')
        else:
            print('Orbit remains bound.',end='\n\n')

        print('New masses:',m_SN-dm_SN,m_comp-dm_comp)
        print('New semi-major axis:',a_new)
        print('New eccentricity:',e_new)
        print('New inclination:',np.arccos(cos_i_new))
        print('New relative velocity:',np.linalg.norm(vvec_new))
        print('New com velocity:',np.linalg.norm(v_com_new),end='\n\n')

        print('Units:',units[0],',',units[1],',',units[2],end='\n\n')

    return a_new,e_new,cos_i_new,Omega_new,omega_new,f_new,v_com_new

def check_triple_stability(a_in,a_out,e_out,m_in,m_out):
    '''
    Input:
        a_in: semi-major axis of inner binary (AU)
        a_out: semi-major axis of outer binary (AU)
        e_out: eccentricity of outer binary
        m_in: mass of inner binary (Msun)
        m_out: mass of tertiary companion (Msun)
    
    Output:
        stable: True if triple is stable, False otherwise
    '''

    # Raise error if input is not a number
    if(not all(isinstance(x,(int,float)) for x in [a_in,a_out,e_out,m_in,m_out])):
        raise ValueError('All input parameters must be numbers')
    
    # Raise error if input is not a number
    if(a_in<0 or a_out<0 or m_in<0 or m_out<0 or e_out<0 or e_out>=1):
        raise ValueError('a_in,a_out,m_in,m_out must be positive and e_out must be between 0 and 1')

    # Check if inner binary is stable
    stable = a_out/a_in > 2.8/(1-e_out)*((m_in+m_out)/m_in*(1+e_out)/np.sqrt(1-e_out))**(2/5)

    return stable

def orbital_period(a,m=1,units=(u.AU,u.yr,u.Msun)):
    '''
    Input:
        a: semi-major axis (AU)
        m: total mass (Msun)
        units: units of the input/output parameters

    Output:
        T: orbital period (yr)
    '''

    # Raise error if input is not a number
    if(not all(isinstance(x,(int,float)) for x in [a,m])):
        raise ValueError('a,m must be numbers')
    
    # Raise error if a is negative
    if(a<0):
        raise ValueError('a must be positive')
    
    # Raise error if m is negative
    if(m<0):
        raise ValueError('m must be positive')

    m *= G.to(units[0]**3/units[1]**2/units[2]).value

    T = 2*np.pi*np.sqrt(a**3/m)

    return T

def semi_major_axis(T,m=1,units=(u.AU,u.yr,u.Msun)):
    '''
    Input:
        T: orbital period (yr)
        m: total mass (Msun)
        units: units of the input/output parameters

    Output:
        a: semi-major axis (AU)
    '''

    # Raise error if input is not a number
    if(not all(isinstance(x,(int,float)) for x in [T,m])):
        raise ValueError('T,m must be numbers')
    
    # Raise error if T is negative
    if(T<0):
        raise ValueError('T must be positive')
    
    # Raise error if m is negative
    if(m<0):
        raise ValueError('m must be positive')

    m *= G.to(units[0]**3/units[1]**2/units[2]).value

    a = (T**2/(4*np.pi**2)*m)**(1/3)

    return a

def Kozai_timescale(a_in,a_out,e_out,m_in,m_out,units=(u.AU,u.yr,u.Msun)):
    '''
    Input:
        a_in: semi-major axis of inner binary (AU)
        a_out: semi-major axis of outer binary (AU)
        e_out: eccentricity of outer binary
        m_in: mass of inner binary (Msun)
        m_out: mass of tertiary companion (Msun)
        units: units of the input/output parameters

    Output:
        T_Kozai: Kozai timescale (yr)
    '''

    # Raise error if input is not a number
    if(not all(isinstance(x,(int,float)) for x in [a_in,a_out,e_out,m_in,m_out])):
        raise ValueError('All input parameters must be numbers')
    
    # Raise error if a_in, a_out, m_in, m_out are negative or if e_out is not between 0 and 1
    if(a_in<0 or a_out<0 or m_in<0 or m_out<0 or e_out<0 or e_out>=1):
        raise ValueError('a_in,a_out,m_in,m_out must be positive and e_out must be between 0 and 1')
    
    m_in *= G.to(units[0]**3/units[1]**2/units[2]).value
    m_out *= G.to(units[0]**3/units[1]**2/units[2]).value

    # Calculate mean motion
    n = np.sqrt(m_in/a_in**3)

    # Calculate Kozai timescale
    T_Kozai = m_in/n/m_out*(a_out/a_in)**3*(1-e_out**2)**(3/2)

    return T_Kozai

def Roche_lobe_radius(m1,m2):
    '''
    Input:
        m1: mass of the primary
        m2: mass of the secondary
    Output:
        R_L: Roche lobe radius (units of separation)
    '''

    # Raise error if input is not a number
    if(not all(isinstance(x,(int,float)) for x in [m1,m2])):
        raise ValueError('m1,m2 must be numbers')
    
    # Raise error if m1, m2 are negative
    if(m1<0 or m2<0):
        raise ValueError('m1,m2 must be positive')
    
    q = m1/m2

    R_L = 0.49*q**(2/3)/(0.6*q**(2/3)+np.log(1+q**(1/3)))

    return R_L
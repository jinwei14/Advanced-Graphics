import matplotlib.pyplot as plt
import math
import numpy as np


def find_brewster(eta_i, eta_t):
    return math.atan(eta_t/eta_i)


def find_critical(eta_i, eta_t):
    return math.asin(eta_t/eta_i)


def cal_fresnel(theta_i, eta_i, eta_t):

    # protection on float precision here
    theta_t = math.asin(min(1., (eta_i/eta_t) * math.sin(theta_i)))
    r_p = (eta_t*math.cos(theta_i) - eta_i*math.cos(theta_t))/(eta_t*math.cos(theta_i) + eta_i*math.cos(theta_t))
    r_p = r_p**2
    r_s = (eta_i*math.cos(theta_i) - eta_t*math.cos(theta_t))/(eta_i*math.cos(theta_i) + eta_t*math.cos(theta_t))
    r_s = r_s**2
    reflectance = (r_p + r_s)/2

    return r_p, r_s, reflectance


def cal_schlick(theta_i, r_0):
    print(math.degrees(math.cos(theta_i)))
    return r_0 + (1.-r_0)*((1. - math.cos(theta_i))**5.)


def plot_fresnel(eta_i, eta_t):

    # set upper bound and calculate reflectance of normal incident
    if eta_i < eta_t:
        upper_bound = 90
        brewster_angle = math.degrees(find_brewster(eta_i, eta_t))
        plt.vlines(brewster_angle, 0., 1., colors='k', linestyles="dashed")
        print('Brewster Angle = ', brewster_angle)
    else:
        upper_bound = math.degrees(find_critical(eta_i, eta_t))
        plt.vlines(upper_bound, 0., 1., colors='k', linestyles="dashed")
        print('Critical Angle = ', upper_bound)

    # init
    r_p_plot = np.ones(100)
    r_s_plot = np.ones(100)
    r = np.ones(100)
    r_sh = np.ones(100)
    x_plot = np.linspace(0, upper_bound, 100)

    r_0 = cal_fresnel(0, eta_i, eta_t)[2]

    for i in range(100):
        # calculate polarized and reflectance of an angle, with its Schlick's approximation
        incident_theta = math.radians(x_plot[i])
        r_p_plot[i], r_s_plot[i], r[i] = cal_fresnel(incident_theta, eta_i, eta_t)
        r_sh[i] = cal_schlick(np.arcsin(max(1, (eta_i/eta_t)) * np.sin(incident_theta)), r_0)

    plt.plot(x_plot, r_p_plot, 'b-', label=u'Parallel Polarized')
    plt.plot(x_plot, r_s_plot, 'y-', label=u'Perpendicular Polarized')
    plt.plot(x_plot, r, 'g-', label=u'Unpolarized')
    plt.plot(x_plot, r_sh, 'r--', label=u'Schlick\'s Approximation')
    plt.title("Eta_i = 1.0, Eta_t = 1.45")
    plt.xlabel('Angles in Degree')
    plt.ylabel('Reflectance')
    plt.legend()
    plt.xlim(0, 90)
    plt.ylim(0, 1)
    plt.show()


plot_fresnel(eta_i=1.45, eta_t=1.)


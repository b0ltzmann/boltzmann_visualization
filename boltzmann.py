#!/usr/bin/env python
import numpy as np
import scipy.constants as pc
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button


class Boltz:
    def __init__(self, model, i=1000, autoscale=False):
        self.model = model
        # Number of energy levels to model:
        self.i = i
        # Autoscale the axes:
        self.autoscale = autoscale
        # Constants:
        self.h = pc.physical_constants["Planck constant"][0]
        self.hb = pc.physical_constants["reduced Planck constant"][0]
        self.kb = pc.physical_constants["Boltzmann constant"][0]
        self.amu = pc.physical_constants["unified atomic mass unit"][0]
        # Initial parameters (F2 as an example):
        self.T0 = 300  # temperature      / K
        self.L0 = 0.5  # box length       / nm
        self.m0 = 38  # mass of particle / amu
        self.kf0 = 445  # spring constant  / Nm-1
        self.mu0 = 9.5  # reduced mass     / amu
        self.r0 = 141  # bond length      / pm

    @property
    def levels(self):
        """
        Generates i quantum numbers (levels) based on the model.
        output:
            levels       (np array)
        """
        if self.model == "PiB":
            levels = np.linspace(1, self.i, num=self.i)
        if self.model == "QMRR":
            levels = np.linspace(0, self.i, num=self.i)
        if self.model == "QMHO":
            levels = np.linspace(0, self.i, num=self.i)
        return levels

    @property
    def degeneracies(self):
        """
        Generates degeneracies of quantum levels based on the model.
        output:
            degeneracies (np array)
        """
        if self.model == "PiB":
            degeneracies = np.ones(len(self.levels))
        if self.model == "QMRR":
            degeneracies = 2 * self.levels + 1
        if self.model == "QMHO":
            degeneracies = np.ones(len(self.levels))
        return degeneracies

    def quantum(self, **kwargs):
        """
        Calculates energies of quantum levels based on the model.
        By statistical mechanics convention, the ground state (zero point)
        energy is set to zero.
        input:
            kwargs - variables depending on model:
                PiB:
                    m: mass of particle / amu
                    L: length of 1D box / nm
                QMHO:
                    mu: reduced mass    / amu
                    kf: force constant  / Nm-1
                QMRR:
                    mu: reduced mass    / amu
                    r: bond length      / pm
        output:
            energies     (np array)
        """
        if self.model == "PiB":
            mass = kwargs.get("m") * self.amu
            length = kwargs.get("L") * 1e-9
            energies = (self.levels**2 * self.h**2) / (8 * mass * length**2)
            energies = energies - energies[0]
        elif self.model == "QMRR":
            moment = kwargs.get("mu") * self.amu * ((kwargs.get("r") * 1e-12) ** 2)
            energies = (self.hb**2 / (2 * moment)) * self.levels * (self.levels + 1)
        elif self.model == "QMHO":
            freq = np.sqrt(kwargs.get("kf") / (kwargs.get("mu") * self.amu))
            energies = (self.levels + 1 / 2) * self.hb * freq
            energies = energies - energies[0]
        else:
            print("Select an appropriate model (PiB, QMRR, or QMHO)")
            quit()
        return energies

    def boltz(self, energies, T):
        """
        Calculates the fractional population of each quantum level.
        input:
            energies     (np array)
            T            (float)
        output:
            fractions    (np array)
            q            (np array)
        """
        boltz = self.degeneracies * np.exp(-energies / (self.kb * T))
        q = boltz.sum()
        fractions = boltz / q
        return fractions, q

    def makeplot(self):
        """
        Creates the main plot that will be updated with sliders.
        """
        self.fig, self.ax = plt.subplots()
        self.fig.subplots_adjust(bottom=0.25)
        self.ax.set_xlabel("Fractional Population")
        self.ax.set_ylabel("Energy Level / J")
        # Text for q:
        self.qtext = self.ax.text(
            0.7, 0.6, "$q=$", size=15, transform=self.ax.transAxes
        )
        # Text for model:
        self.ax.text(0.7, 0.9, self.model, size=15, transform=self.ax.transAxes)
        # Sliders locations:
        axtemp = plt.axes([0.25, 0.15, 0.60, 0.03])
        axmass = plt.axes([0.25, 0.10, 0.60, 0.03])
        axbox = plt.axes([0.25, 0.05, 0.60, 0.03])
        # All models have a temperature slider:
        self.temp_slider = Slider(
            ax=axtemp,
            label="Temperature, $T$ / K",
            valmin=0.1,
            valmax=2000,
            valinit=self.T0,
        )
        # Specific slider settings and equations:
        if self.model == "PiB":
            self.mass_slider = Slider(
                ax=axmass,
                label="Mass, $m$ / amu",
                valmin=1,
                valmax=100,
                valinit=self.m0,
            )
            self.box_slider = Slider(
                ax=axbox,
                label="Length, $L$ / nm",
                valmin=0.1,
                valmax=10,
                valinit=self.L0,
            )
            self.ax.text(
                0.7,
                0.8,
                r"$E_n=\frac{n^2h^2}{8mL^2}$",
                size=15,
                transform=self.ax.transAxes,
            )
            self.ax.text(
                0.7, 0.7, r"$g_n=1$ $\forall$ $n$", size=15, transform=self.ax.transAxes
            )
        if self.model == "QMRR":
            self.mass_slider = Slider(
                ax=axmass,
                label="Reduced Mass, $\mu$ / amu",
                valmin=1,
                valmax=100,
                valinit=self.mu0,
            )
            self.box_slider = Slider(
                ax=axbox,
                label="Bond Length, $r$ / pm",
                valmin=50,
                valmax=300,
                valinit=self.r0,
            )
            self.ax.text(
                0.7,
                0.8,
                r"$E_J=\frac{\hbar^2}{2\mu r^2}J(J+1)$",
                size=15,
                transform=self.ax.transAxes,
            )
            self.ax.text(0.7, 0.7, r"$g_J=2J+1$", size=15, transform=self.ax.transAxes)
        if self.model == "QMHO":
            self.mass_slider = Slider(
                ax=axmass,
                label="Reduced Mass, $\mu$ / amu",
                valmin=1,
                valmax=100,
                valinit=self.mu0,
            )
            self.box_slider = Slider(
                ax=axbox,
                label="Force Constant, $k_f$ / N/m",
                valmin=100,
                valmax=3000,
                valinit=self.kf0,
            )
            self.ax.text(
                0.7,
                0.8,
                r"$E_v=(v+0.5)\hbar\sqrt{\frac{k_f}{\mu}}$",
                size=15,
                transform=self.ax.transAxes,
            )
            self.ax.text(
                0.7, 0.7, r"$g_v=1$ $\forall$ $v$", size=15, transform=self.ax.transAxes
            )

    def addtoplot(self, energies, fractions, q, T):
        """
        Adds to plot:
            1. current q value
            2. kBT line visualizing average energy
            3. horizontal lines to visualize the population of states:
                Generates two lists to add i two point lines to a plot.
                Each pair of points is separated by None.
                Significantly faster slider performance than ax.hlines.
        """
        # Remove current lines:
        for line in self.ax.get_lines():
            line.remove()
        # Update value for q:
        self.qtext.set_text("$q=${0:.1f}".format(q))
        # Add kBT line
        self.ax.plot((0, 1), (self.kb * T, self.kb * T), lw=2, ls=":", color="#cc3333")
        # Generate two lists for horizontal lines:
        xlist = []
        ylist = []
        for i in range(0, len(energies), 1):
            xlist.extend([0, fractions[i], None])
            ylist.extend([energies[i], energies[i], None])
        self.ax.plot(xlist, ylist, lw=2, color="#336699")
        if self.autoscale:
            # Set xmax 0.01 greater than largest frac pop (skip None's for max)
            self.ax.set_xlim((0, max(xlist[1::3]) + 0.01))
            # Set ymax to show levels with frac pop > 0.001:
            maxpop = np.where(fractions >= 0.001)[0][-1]
            self.ax.set_ylim((0, energies[maxpop]))

    def update(self, val):
        """
        Recalculates energies and fractional populations from slider values
        and updates the plot.
        """
        if self.model == "PiB":
            energies = self.quantum(m=self.mass_slider.val, L=self.box_slider.val)
        if self.model == "QMRR":
            energies = self.quantum(mu=self.mass_slider.val, r=self.box_slider.val)
        if self.model == "QMHO":
            energies = self.quantum(mu=self.mass_slider.val, kf=self.box_slider.val)
        fractions, q = self.boltz(energies, self.temp_slider.val)
        self.addtoplot(energies, fractions, q, self.temp_slider.val)

    def initialize(self):
        """
        Calculates energies and fractional populations from initial values,
        creates/displays the plot, adds to the plot, and sets up the sliders.
        """
        energies = self.quantum(
            m=self.m0, L=self.L0, kf=self.kf0, mu=self.mu0, r=self.r0
        )
        fractions, q = self.boltz(energies, self.T0)
        self.makeplot()
        # Set reasonable axis limits if autoscale is False:
        if not self.autoscale:
            if self.model == "PiB":
                self.ax.set_xlim((0, 0.05))
                self.ax.set_ylim((0, energies[100]))
            if self.model == "QMRR":
                self.ax.set_xlim((0, 0.10))
                self.ax.set_ylim((0, energies[50]))
            if self.model == "QMHO":
                self.ax.set_xlim((0, 0.5))
                self.ax.set_ylim((0, energies[5]))
        self.addtoplot(energies, fractions, q, self.T0)
        self.temp_slider.on_changed(self.update)
        self.mass_slider.on_changed(self.update)
        self.box_slider.on_changed(self.update)
        plt.show()


if __name__ == "__main__":
    model = str(input("Which model? (PiB, QMRR, or QMHO)\n"))
    scale = str(input("Autoscale axes? (y/N)\n"))
    if scale in ["y", "Y", "Yes", "yes"]:
        visualize = Boltz(model, autoscale=True)
    else:
        visualize = Boltz(model)
    visualize.initialize()

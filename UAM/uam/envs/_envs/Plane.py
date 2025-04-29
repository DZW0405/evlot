import json
import math
from copy import copy

import bluesky as bs
import bluesky.tools.aero as aero
import bluesky.traffic.performance.openap.thrust as thrust
import numpy as np
import pandas as pd
import utm
from bluesky.tools.geo import latlondist

bs.settings.set_variable_defaults(perf_path_openap="performance/OpenAP")

LIFT_FIXWING = 1  # fixwing aircraft
LIFT_ROTOR = 2  # rotor aircraft

ENG_TYPE_TF = 1  # turbofan, fixwing
ENG_TYPE_TP = 2  # turboprop, fixwing
ENG_TYPE_TS = 3  # turboshlft, rotor

NA = 0  # Unknown phase
GD = 1  # Ground
IC = 2  # Initial climb
CL = 3  # Climb
CR = 4  # Cruise
DE = 5  # Descent
AP = 6  # Approach

def angle_to_2pi(angle):
    if angle >= 0:
        return angle - math.floor(angle / (2*math.pi)) * 2*math.pi
    else:
        return angle + (math.floor(-angle / (2*math.pi)) + 1) * 2*math.pi
# 5.将角度转换为[-pi， pi）
def angle_to_pi(angle):
    if angle >= 0:
        return angle - math.floor((angle + math.pi) / (2*math.pi)) * 2*math.pi
    else:
        return angle + math.floor((-angle + math.pi)  / (2*math.pi)) * 2*math.pi
# 6. 相对方位
def bng_rel(N0, E0, N1, E1, head0, to_2pi=True):
    """计算相对方位(in radiant in [0, 2pi) if to_2pi, else [-pi, pi)) of
    (N1, E1) from perspective of (N0, E0) and heading head0."""
    if to_2pi:
        return angle_to_2pi(bng_abs(N0, E0, N1, E1) - head0)
    else:
        return angle_to_pi(bng_abs(N0, E0, N1, E1) - head0)

def meter_to_NM(meter):
    return meter / 1852

def polar_from_xy(x, y, with_r=True, with_angle=True):
    """Get polar coordinates (r, angle in rad in [0, 2pi)) from x,y-coordinates. Angles are defined clockwise with zero at the y-axis.
    Args:
        with_r (bool):     Whether to compute the radius.
        with_angle (bool): Whether to compute the angle.
    Returns:
        r, angle as a tuple of floats."""

    r = math.sqrt(x**2 + y**2) if with_r else None
    angle = angle_to_2pi(math.atan2(x, y)) if with_angle else None
    return r, angle

def cte(N1, E1, N2, E2, NA, EA, pi_path=None):
    """Computes the cross-track error following p.350 of Fossen (2021). The waypoints are (N1, E1) and (N2, E2), while
    the agent is at (NA, EA). The angle of the path relative to the NED-frame can optionally be specified."""
    if pi_path is None:
        pi_path = bng_abs(N0=N1, E0=E1, N1=N2, E1=E2)
    return -np.sin(pi_path)*(NA - N1) + np.cos(pi_path)*(EA - E1)

def ate(N1, E1, N2, E2, NA, EA, pi_path=None):
    """Computes the along-track error following p.350 of Fossen (2021). The waypoints are (N1, E1) and (N2, E2), while
    the agent is at (NA, EA). The angle of the path relative to the NED-frame can optionally be specified."""
    if pi_path is None:
        pi_path = bng_abs(N0=N1, E0=E1, N1=N2, E1=E2)
    return np.cos(pi_path)*(NA - N1) + np.sin(pi_path)*(EA - E1)

def bng_abs(N0, E0, N1, E1):
    """Computes the absolute bearing (in radiant, [0, 2pi)) of (N1, E1) from perspective of (N0, E0)."""
    return polar_from_xy(x=E1-E0, y=N1-N0, with_r=False, with_angle=True)[1]

def ED(N0, E0, N1, E1, sqrt=True):
    d_sq = (N0 - N1)**2 + (E0 - E1)**2

    if sqrt:
        return np.sqrt(d_sq)
    return d_sq

def to_latlon(north, east, number):
    """Converts North, East, number in UTM into longitude and latitude. Assumes northern hemisphere.
    Returns: (lat, lon)"""
    if isinstance(north, list):
        north = np.array(north)
    if isinstance(east, list):
        east = np.array(east)
    return utm.to_latlon(easting=east, northing=north, zone_number=number, northern=True, strict=False)

def xy_from_polar(r, angle):
    return r * np.sin(angle), r * np.cos(angle)

# 2.dtr
def dtr(angle):
    return angle * math.pi / 180

# 3.tcpa
def tcpa(NOS, EOS, NTS, ETS, chiOS, chiTS, VOS, VTS):
    """计算最接近点的时间."""
    # easy access
    xOS,yOS,xTS,yTS = EOS,NOS,ETS,NTS
    # 以x，y坐标计算速度
    vxOS, vyOS = xy_from_polar(r=VOS, angle=chiOS)
    vxTS, vyTS = xy_from_polar(r=VTS, angle=chiTS)
    # 相对速度
    vrx = vxTS - vxOS
    vry = vyTS - vyOS
    # tcpa
    nom = (xTS - xOS)*vrx + (yTS - yOS)*vry
    den = vrx**2 + vry**2

    if den == 0:
        return 0.0
    return - nom / den

# 4. cpa
def cpa(NOS, EOS, NTS, ETS, chiOS, chiTS, VOS, VTS, get_positions=False):
    # easy access
    xOS, yOS, xTS, yTS = EOS, NOS, ETS, NTS

    # 以x,y坐标计算速度
    vxOS, vyOS = xy_from_polar(r=VOS, angle=chiOS)
    vxTS, vyTS = xy_from_polar(r=VTS, angle=chiTS)

    # get TCPA
    TCPA = tcpa(NOS, EOS, NTS, ETS, chiOS, chiTS, VOS, VTS)

    # 预测 OS，TS
    xOS_tcpa = xOS + TCPA * vxOS
    yOS_tcpa = yOS + TCPA * vyOS
    xTS_tcpa = xTS + TCPA * vxTS
    yTS_tcpa = yTS + TCPA * vyTS

    if get_positions:
        return ED(N0=yOS_tcpa, E0=xOS_tcpa, N1=yTS_tcpa, E1=xTS_tcpa), TCPA, yOS_tcpa, xOS_tcpa, yTS_tcpa, xTS_tcpa
    else:
        return ED(N0=yOS_tcpa, E0=xOS_tcpa, N1=yTS_tcpa, E1=xTS_tcpa), TCPA

def to_utm(lat, lon):
    """Converts latitude and longitude into North, East, and zone number in UTM.
    Returns: (North, East, number)."""
    E, N, number, _ = utm.from_latlon(latitude=lat, longitude=lon)
    return (N, E, number)

def rtd(angle):
    """Takes angle in radiant and transforms it to degree."""
    return angle * 180 / math.pi

def prep_angles_for_average(ang1, ang2):
    """Prepares two angles before they get averaged. Necessary since otherwise there are boundary issues at 2pi."""
    # make sure they are in [0,2pi)
    ang1 = angle_to_2pi(ang1)
    ang2 = angle_to_2pi(ang2)

    # transform if necessary
    if abs(ang1-ang2) >= math.pi:
        if ang1 >= ang2:
            ang1 = angle_to_pi(ang1)
        else:
            ang2 = angle_to_pi(ang2)
    return ang1, ang2

def VFG(N1, E1, N2, E2, NA, EA, K, N3=None, E3=None):
    """Computes the cte, desired course, and path angle based on the vector field guidance method following pp.354-55 of Fossen (2021).
    The waypoints are (N1, E1) and (N2, E2), while the agent is at (NA, EA). K is the convergence rate of the vector field.

    Returns:
        cte (float), desired_course (float, rad), path_angle12 (float, rad), smoothed path_angle (float, rad)"""

    assert K > 0, "K of VFG must be larger zero."

    # get path angle between point 1 and 2
    pi_path_12 = bng_abs(N0=N1, E0=E1, N1=N2, E1=E2)

    # get CTE
    ye = cte(N1=N1, E1=E1, N2=N2, E2=E2, NA=NA, EA=EA, pi_path=pi_path_12)

    # potentially consired point 3 for desired course
    if N3 is not None and E3 is not None:

        # ate between point 1 and 2
        ate_12 = ate(N1=N1, E1=E1, N2=N2, E2=E2, NA=NA, EA=EA, pi_path=pi_path_12)

        # path angle between points 2 and 3
        pi_path_23 = bng_abs(N0=N2, E0=E2, N1=N3, E1=E3)

        # percentage of overall distance
        frac = np.clip(ate_12 / ED(N0=N1, E0=E1, N1=N2, E1=E2), 0.0, 1.0)
        w23 = frac ** 1

        # adjustment to avoid boundary issues at 2pi
        pi_path_12, pi_path_23 = prep_angles_for_average(pi_path_12, pi_path_23)

        # construct new angle
        pi_path_up = angle_to_2pi(w23 * pi_path_23 + (1 - w23) * pi_path_12)

    else:
        pi_path_up = pi_path_12

    # get desired course, which is rotated back to NED
    return ye, angle_to_2pi(pi_path_up - math.atan(ye * K)), pi_path_12, pi_path_up

def switch_wp(wp1_N, wp1_E, wp2_N, wp2_E, a_N, a_E):
    pi_path = bng_abs(N0=wp1_N, E0=wp1_E, N1=wp2_N, E1=wp2_E)

    # check relative bearing
    bng_rel_p = angle_to_pi(bng_rel(N0=a_N, E0=a_E, N1=wp2_N, E1=wp2_E, head0=pi_path))

    if abs(rtd(bng_rel_p)) > 90:
        return True
    else:
        return False


def get_init_two_wp(n_array, e_array, a_n, a_e, stop_goal=False):
    EDs = ED(N0=a_n, E0=a_e, N1=n_array, E1=e_array, sqrt=False)
    min_idx = np.argmin(EDs)

    # limit cases
    if min_idx in [0, len(n_array) - 1]:

        # start of path
        if min_idx == 0:
            idx1 = 0
            idx2 = 1

        # end of path
        else:
            idx1 = len(n_array) - 2
            idx2 = len(n_array) - 1
            if stop_goal:
                raise ValueError("The agent spawned already at the goal!")

        wp1_N = n_array[idx1]
        wp1_E = e_array[idx1]
        wp2_N = n_array[idx2]
        wp2_E = e_array[idx2]

    # arbitrarily select prior index as current set of wps
    else:
        idx1 = min_idx - 1
        idx2 = min_idx
        wp1_N = n_array[idx1]
        wp1_E = e_array[idx1]
        wp2_N = n_array[idx2]
        wp2_E = e_array[idx2]

        # check whether waypoints should be switched
        if switch_wp(wp1_N=wp1_N, wp1_E=wp1_E, wp2_N=wp2_N, wp2_E=wp2_E, a_N=a_n, a_E=a_e):
            idx1 += 1
            idx2 += 1
            wp1_N = n_array[idx1]
            wp1_E = e_array[idx1]
            wp2_N = n_array[idx2]
            wp2_E = e_array[idx2]
    return idx1, wp1_N, wp1_E, idx2, wp2_N, wp2_E

class Coefficient:
    def __init__(self):
        # Load synonyms.dat text file into dictionary
        self.synodict = {}
        with open(bs.resource(bs.settings.perf_path_openap) / 'synonym.dat', "r") as f_syno:
            for line in f_syno.readlines():
                if line.count("#") > 0:
                    dataline, comment = line.split("#")
                else:
                    dataline = line.strip("\n")
                acmod, synomod = dataline.split("=")
                acmod = acmod.strip().upper()
                synomod = synomod.strip().upper()

                if acmod == synomod:
                    continue
                self.synodict[acmod] = synomod

        self.acs_fixwing = self._load_all_fixwing_flavor()
        self.engines_fixwing = pd.read_csv(bs.resource(bs.settings.perf_path_openap) / "fixwing/engines.csv", encoding="utf-8")
        self.limits_fixwing = self._load_all_fixwing_envelop()

        self.acs_rotor = self._load_all_rotor_flavor()
        self.limits_rotor = self._load_all_rotor_envelop()

        self.actypes_fixwing = list(self.acs_fixwing.keys())
        self.actypes_rotor = list(self.acs_rotor.keys())

        df = pd.read_csv(bs.resource(bs.settings.perf_path_openap) / "fixwing/dragpolar.csv", index_col="mdl")
        self.dragpolar_fixwing = df.to_dict(orient="index")
        self.dragpolar_fixwing["NA"] = df.mean().to_dict()

    def _load_all_fixwing_flavor(self):
        import warnings

        warnings.simplefilter("ignore")

        # read fixwing aircraft and engine files
        allengines = pd.read_csv(bs.resource(bs.settings.perf_path_openap) / "fixwing/engines.csv", encoding="utf-8")
        allengines["name"] = allengines["name"].str.upper()
        acs = json.load(open(bs.resource(bs.settings.perf_path_openap) / "fixwing/aircraft.json", "r"))
        acs.pop("__comment")
        acs_ = {}

        for mdl, ac in acs.items():
            acengines = ac["engines"]
            acs_[mdl.upper()] = ac.copy()
            acs_[mdl.upper()]["lifttype"] = LIFT_FIXWING
            acs_[mdl.upper()]["engines"] = {}

            for e in acengines:
                e = e.strip().upper()
                selengine = allengines[allengines["name"].str.startswith(e)]
                if selengine.shape[0] >= 1:
                    engine = json.loads(selengine.iloc[-1, :].to_json())
                    acs_[mdl.upper()]["engines"][engine["name"]] = engine

        return acs_

    def _load_all_rotor_flavor(self):
        # read rotor aircraft
        acs = json.load(open(bs.resource(bs.settings.perf_path_openap) / "rotor/aircraft.json", "r"))
        acs.pop("__comment")
        acs_ = {}
        for mdl, ac in acs.items():
            acs_[mdl.upper()] = ac.copy()
            acs_[mdl.upper()]["lifttype"] = LIFT_ROTOR
        return acs_

    def _load_all_fixwing_envelop(self):
        """load aircraft envelop from the model database,
        All unit in SI"""
        limits_fixwing = {}
        for mdl, ac in self.acs_fixwing.items():
            fenv = bs.resource(bs.settings.perf_path_openap) / "fixwing/wrap" / (mdl.lower() + ".txt")

            if fenv.is_file():
                df = pd.read_fwf(fenv).set_index("variable")
                limits_fixwing[mdl] = {}
                limits_fixwing[mdl]["vminto"] = df.loc["to_v_lof"]["min"]
                limits_fixwing[mdl]["vmaxto"] = df.loc["to_v_lof"]["max"]
                limits_fixwing[mdl]["vminic"] = df.loc["ic_va_avg"]["min"]
                limits_fixwing[mdl]["vmaxic"] = df.loc["ic_va_avg"]["max"]
                limits_fixwing[mdl]["vminer"] = min(
                    df.loc["ic_va_avg"]["min"],
                    df.loc["cl_v_cas_const"]["min"],
                    df.loc["cr_v_cas_mean"]["min"],
                    df.loc["de_v_cas_const"]["min"],
                    df.loc["fa_va_avg"]["min"],
                )
                limits_fixwing[mdl]["vmaxer"] = max(
                    df.loc["ic_va_avg"]["max"],
                    df.loc["cl_v_cas_const"]["max"],
                    df.loc["cr_v_cas_mean"]["max"],
                    df.loc["de_v_cas_const"]["max"],
                    df.loc["fa_va_avg"]["max"],
                )
                limits_fixwing[mdl]["vminap"] = df.loc["fa_va_avg"]["min"]
                limits_fixwing[mdl]["vmaxap"] = df.loc["fa_va_avg"]["max"]
                limits_fixwing[mdl]["vminld"] = df.loc["ld_v_app"]["min"]
                limits_fixwing[mdl]["vmaxld"] = df.loc["ld_v_app"]["max"]

                limits_fixwing[mdl]["vmo"] = limits_fixwing[mdl]["vmaxer"]
                limits_fixwing[mdl]["mmo"] = df.loc["cr_v_mach_max"]["opt"]

                limits_fixwing[mdl]["hmax"] = df.loc["cr_h_max"]["opt"] * 1000
                limits_fixwing[mdl]["crosscl"] = df.loc["cl_h_mach_const"]["opt"]
                limits_fixwing[mdl]["crossde"] = df.loc["de_h_cas_const"]["opt"]

                limits_fixwing[mdl]["axmax"] = df.loc["to_acc_tof"]["max"]

                limits_fixwing[mdl]["vsmax"] = max(
                    df.loc["ic_vs_avg"]["max"],
                    df.loc["cl_vs_avg_pre_cas"]["max"],
                    df.loc["cl_vs_avg_cas_const"]["max"],
                    df.loc["cl_vs_avg_mach_const"]["max"],
                )

                limits_fixwing[mdl]["vsmin"] = min(
                    df.loc["ic_vs_avg"]["min"],
                    df.loc["de_vs_avg_after_cas"]["min"],
                    df.loc["de_vs_avg_cas_const"]["min"],
                    df.loc["de_vs_avg_mach_const"]["min"],
                )

        # create envolop based on synonym
        for mdl in self.synodict.keys():
            if mdl not in limits_fixwing:
                limits_fixwing[mdl] = limits_fixwing[self.synodict[mdl]]

        return limits_fixwing

    def _load_all_rotor_envelop(self):
        """load rotor aircraft envelop, all unit in SI"""
        limits_rotor = {}
        for mdl, ac in self.acs_rotor.items():
            limits_rotor[mdl] = {}

            limits_rotor[mdl]["vmin"] = ac["envelop"].get("v_min", -20)
            limits_rotor[mdl]["vmax"] = ac["envelop"].get("v_max", 20)
            limits_rotor[mdl]["vsmin"] = ac["envelop"].get("vs_min", -5)
            limits_rotor[mdl]["vsmax"] = ac["envelop"].get("vs_max", 5)
            limits_rotor[mdl]["hmax"] = ac["envelop"].get("h_max", 2500)

            params = ["v_min", "v_max", "vs_min", "vs_max", "h_max"]
            if set(params) <= set(ac["envelop"].keys()):
                pass
            else:
                warn = f"Warning: Some performance parameters for {mdl} are not found, default values used."
                print(warn)

        return limits_rotor

class OpenAP:
    """
    Open-source Aircraft Performance (OpenAP) Model

    Methods:
        create(): initialize new aircraft with performance parameters
        update(): update performance parameters
    """
    def __init__(self, actype, tas, alt):
        self.ac_warning = False   # aircraft mdl to default warning
        self.eng_warning = False  # aircraft engine to default warning
        self.coeff = Coefficient()

        # check synonym file if not in open ap actypes
        if (actype not in self.coeff.actypes_rotor) and (
            actype not in self.coeff.dragpolar_fixwing
        ):
            if actype in self.coeff.synodict.keys():
                actype = self.coeff.synodict[actype]

        # initialize aircraft / engine performance parameters
        # check fixwing or rotor, default to fixwing
        if actype in self.coeff.actypes_rotor:
            self.lifttype = LIFT_ROTOR
            self.mass = 0.5 * (
                self.coeff.acs_rotor[actype]["oew"]
                + self.coeff.acs_rotor[actype]["mtow"]
            )
            self.engnum = int(self.coeff.acs_rotor[actype]["n_engines"])
            self.engpower = self.coeff.acs_rotor[actype]["engines"][0][1]

        else:
            # convert to known aircraft type
            if actype not in self.coeff.actypes_fixwing:
                assert ValueError("Aircraft type unknown!")

            # populate fuel flow model
            es = self.coeff.acs_fixwing[actype]["engines"]
            e = es[list(es.keys())[0]]
            coeff_a, coeff_b, coeff_c = thrust.compute_eng_ff_coeff(
                e["ff_idl"], e["ff_app"], e["ff_co"], e["ff_to"]
            )

            self.lifttype = LIFT_FIXWING

            self.Sref = self.coeff.acs_fixwing[actype]["wa"]
            self.mass = 0.5 * (
                self.coeff.acs_fixwing[actype]["oew"]
                + self.coeff.acs_fixwing[actype]["mtow"]
            )

            self.engnum = int(self.coeff.acs_fixwing[actype]["n_engines"])

            self.ff_coeff_a = coeff_a
            self.ff_coeff_b = coeff_b
            self.ff_coeff_c = coeff_c

            all_ac_engs = list(self.coeff.acs_fixwing[actype]["engines"].keys())
            self.engthrmax = self.coeff.acs_fixwing[actype]["engines"][
                all_ac_engs[0]
            ]["thr"]
            self.engbpr = self.coeff.acs_fixwing[actype]["engines"][
                all_ac_engs[0]
            ]["bpr"]

        # init type specific coefficients for flight envelops
        if actype in self.coeff.limits_rotor.keys():  # rotorcraft
            self.vmin = self.coeff.limits_rotor[actype]["vmin"]
            self.vmax = self.coeff.limits_rotor[actype]["vmax"]
            self.vsmin = self.coeff.limits_rotor[actype]["vsmin"]
            self.vsmax = self.coeff.limits_rotor[actype]["vsmax"]
            self.hmax = self.coeff.limits_rotor[actype]["hmax"]

            self.vsmin = self.coeff.limits_rotor[actype]["vsmin"]
            self.vsmax = self.coeff.limits_rotor[actype]["vsmax"]
            self.hmax = self.coeff.limits_rotor[actype]["hmax"]

            self.cd0_clean = np.nan
            self.k_clean = np.nan
            self.cd0_to = np.nan
            self.k_to = np.nan
            self.cd0_ld = np.nan
            self.k_ld = np.nan
            self.delta_cd_gear = np.nan

        else:
            if actype not in self.coeff.limits_fixwing.keys():
                assert ValueError("Aircraft type unknown!")

            self.vminic = self.coeff.limits_fixwing[actype]["vminic"]
            self.vminer = self.coeff.limits_fixwing[actype]["vminer"]
            self.vminap = self.coeff.limits_fixwing[actype]["vminap"]
            self.vmaxic = self.coeff.limits_fixwing[actype]["vmaxic"]
            self.vmaxer = self.coeff.limits_fixwing[actype]["vmaxer"]
            self.vmaxap = self.coeff.limits_fixwing[actype]["vmaxap"]

            self.vsmin = self.coeff.limits_fixwing[actype]["vsmin"]
            self.vsmax = self.coeff.limits_fixwing[actype]["vsmax"]
            self.hmax = self.coeff.limits_fixwing[actype]["hmax"]
            self.axmax = self.coeff.limits_fixwing[actype]["axmax"]
            self.vminto = self.coeff.limits_fixwing[actype]["vminto"]
            self.hcross = self.coeff.limits_fixwing[actype]["crosscl"]
            self.mmo = self.coeff.limits_fixwing[actype]["mmo"]

            self.cd0_clean = self.coeff.dragpolar_fixwing[actype]["cd0_clean"]
            self.k_clean = self.coeff.dragpolar_fixwing[actype]["k_clean"]
            self.cd0_to = self.coeff.dragpolar_fixwing[actype]["cd0_to"]
            self.k_to = self.coeff.dragpolar_fixwing[actype]["k_to"]
            self.cd0_ld = self.coeff.dragpolar_fixwing[actype]["cd0_ld"]
            self.k_ld = self.coeff.dragpolar_fixwing[actype]["k_ld"]
            self.delta_cd_gear = self.coeff.dragpolar_fixwing[actype][
                "delta_cd_gear"
            ]

        # append update actypes, after removing unknown types
        self.actype = actype

        # get phase
        self.phase = self._get_phase(spd=tas, roc=0.0, alt=alt)

        # Update envelope speed limits
        self.vmin, self.vmax = self._construct_v_limits()

    def _get_phase(self, spd, roc, alt, unit="SI"):
        if self.lifttype == LIFT_FIXWING:
            return self._get_fixwing(spd, roc, alt, unit)

        elif self.lifttype == LIFT_ROTOR:
            return self._get_rotor(spd, roc, alt, unit)

    def _get_fixwing(self, spd, roc, alt, unit="SI"):
        if unit not in ["SI", "EP"]:
            raise RuntimeError("wrong unit type")

        if unit == "SI":
            spd = spd / 0.514444
            roc = roc / 0.00508
            alt = alt / 0.3048

        if alt <= 75:
            return GD
        elif (alt >= 75) & (alt <= 1000) & (roc >= 150):
            return IC
        elif (alt >= 75) & (alt <= 1000) & (roc <= -150):
            return AP
        elif (alt >= 1000) & (roc >= 150):
            return CL
        elif (alt >= 1000) & (roc <= -150):
            return DE
        elif (alt >= 10000) & (roc <= 150) & (roc >= -150):
            return CR
        else:
            return NA

    def _get_rotor(self, spd, roc, alt, unit="SI"):
        return NA

    def update(self, tas, vs, alt, ax):
        """Periodic update function for performance calculations."""
        self.tas = tas
        self.vs  = vs
        self.alt = alt
        self.ax  = ax
        
        # update phase, infer from spd, roc, alt
        self.phase = self._get_phase(self.tas, self.vs, self.alt, unit="SI")

        # update speed limits, based on phase change
        self.vmin, self.vmax = self._construct_v_limits()

        if self.lifttype == LIFT_FIXWING:
            # ----- compute drag -----
            # update drage coefficient based on flight phase
            if self.phase == GD:
                self.cd0 = self.cd0_to + self.delta_cd_gear
                self.k = self.k_to

            elif self.phase == IC:
                self.cd0 = self.cd0_to
                self.k = self.k_to

            elif self.phase == AP:
                self.cd0 = self.cd0_ld
                self.k = self.k_ld

            elif self.phase in [CL, CR, DE, NA]:
                self.cd0 = self.cd0_clean
                self.k = self.k_clean

            rho = aero.vdensity(self.alt)
            rhovs = 0.5 * rho * self.tas ** 2 * self.Sref
            cl = self.mass * aero.g0 / rhovs
            self.drag = rhovs * (self.cd0 + self.k * cl ** 2)

            # ----- compute maximum thrust -----
            max_thrustratio_fixwing = self._compute_max_thr_ratio(
                self.phase,
                self.engbpr,
                self.tas,
                self.alt,
                self.vs,
                self.engnum * self.engthrmax,
            )
            self.max_thrust = (
                max_thrustratio_fixwing
                * self.engnum
                * self.engthrmax
            )

            # ----- compute net thrust -----
            self.thrust = (self.drag + self.mass * self.ax)

            # ----- compute duel flow -----
            thrustratio_fixwing = self.thrust / (self.engnum * self.engthrmax)
            self.fuelflow = self.engnum * (
                self.ff_coeff_a * thrustratio_fixwing ** 2
                + self.ff_coeff_b * thrustratio_fixwing
                + self.ff_coeff_c
            )

        # ----- update max acceleration ----
        self.axmax = self.calc_axmax()

        # TODO: implement thrust computation for rotor aircraft

    def _compute_max_thr_ratio(self, phase, bpr, v, h, vs, thr0):
        ratio_takeoff = thrust.tr_takeoff(bpr, v, h)

        # ---- thrust ratio in flight ----
        ratio_inflight = self._tr_inflight(v, h, vs, thr0)

        # thrust ratio array
        #   LD and GN assume ZERO thrust
        if phase == GD:
            return ratio_takeoff
        else:
            return ratio_inflight

    def _tr_inflight(self, v, h, vs, thr0):
        """Compute thrust ration for inflight"""

        def dfunc(mratio):
            d = -0.4204 * mratio + 1.0824
            return d

        def nfunc(roc):
            n = 2.667e-05 * roc + 0.8633
            return n

        def mfunc(vratio, roc):
            m = -1.2043e-1 * vratio - 8.8889e-9 * roc ** 2 + 2.4444e-5 * roc + 4.7379e-1
            return m

        roc = abs(vs / aero.fpm)
        if v < 10:
            v = 10
        mach = aero.vtas2mach(v, h)
        vcas = aero.vtas2cas(v, h)

        p = aero.vpressure(h)
        p10 = aero.vpressure(10000 * aero.ft)
        p35 = aero.vpressure(35000 * aero.ft)

        # approximate thrust at top of climb (REF 2)
        F35 = (200 + 0.2 * thr0 / 4.448) * 4.448
        mach_ref = 0.8
        vcas_ref = aero.vmach2cas(mach_ref, 35000 * aero.ft)

        # segment 3: alt > 35000:
        d = dfunc(mach / mach_ref)
        b = (mach / mach_ref) ** (-0.11)
        ratio_seg3 = d * np.log(p / p35) + b

        # segment 2: 10000 < alt <= 35000:
        a = (vcas / vcas_ref) ** (-0.1)
        n = nfunc(roc)
        ratio_seg2 = a * (p / p35) ** (-0.355 * (vcas / vcas_ref) + n)

        # segment 1: alt <= 10000:
        F10 = F35 * a * (p10 / p35) ** (-0.355 * (vcas / vcas_ref) + n)
        m = mfunc(vcas / vcas_ref, roc)
        ratio_seg1 = m * (p / p35) + (F10 / F35 - m * (p10 / p35))

        if h > 35000 * aero.ft:
            ratio = ratio_seg3
        elif h > 10000 * aero.ft:
            ratio = ratio_seg2
        else:
            ratio = ratio_seg1

        # convert to maximum static thrust ratio
        ratio_F0 = ratio * F35 / thr0
        return ratio_F0

    def limits(self, intent_v_tas, intent_vs, intent_h, ax):
        allow_h = np.where(intent_h > self.hmax, self.hmax, intent_h)

        if self.lifttype == LIFT_FIXWING:
            intent_v_cas = aero.vtas2cas(intent_v_tas, allow_h)
            allow_v_cas = np.where((intent_v_cas < self.vmin), self.vmin, intent_v_cas)
            allow_v_cas = np.where(intent_v_cas > self.vmax, self.vmax, allow_v_cas)
            allow_v_tas = aero.vcas2tas(allow_v_cas, allow_h)
            allow_v_tas = np.where(
                aero.vtas2mach(allow_v_tas, allow_h) > self.mmo,
                aero.vmach2tas(self.mmo, allow_h),
                allow_v_tas,
            )  # maximum cannot exceed MMO

            vs_max_with_acc = (1 - ax / self.axmax) * self.vsmax
            allow_vs = np.where(
                (intent_vs > 0) & (intent_vs > self.vsmax), vs_max_with_acc, intent_vs
            )  # for climb with vs larger than vsmax
            allow_vs = np.where(
                (intent_vs < 0) & (intent_vs < self.vsmin), vs_max_with_acc, allow_vs
            )  # for descent with vs smaller than vsmin (negative)
            allow_vs = np.where(
                (self.phase == GD) & (self.tas < self.vminto), 0, allow_vs
            )  # takeoff aircraft

        # correct rotercraft speed limits
        elif self.lifttype == LIFT_ROTOR:
            allow_v_tas = np.where(
                (intent_v_tas < self.vmin), self.vmin, intent_v_tas
            )
            allow_v_tas = np.where(
                (intent_v_tas > self.vmax), self.vmax, allow_v_tas
            )
            allow_vs = np.where(
                (intent_vs < self.vsmin), self.vsmin, intent_vs
            )
            allow_vs = np.where(
                (intent_vs > self.vsmax), self.vsmax, allow_vs
            )
        return float(allow_v_tas), float(allow_vs), float(allow_h)

    def currentlimits(self):
        vtasmin = aero.vcas2tas(self.vmin, self.alt)
        vtasmax = min([aero.vcas2tas(self.vmax, self.alt), aero.vmach2tas(self.mmo, self.alt)])
        return vtasmin, vtasmax, self.vsmin, self.vsmax

    def _construct_v_limits(self):
        if self.lifttype == LIFT_FIXWING:

            # --- minimum and minimum speed ---
            if self.phase == NA:
                vmin = 0.0
                vmax = self.vmaxer
            elif self.phase == IC:
                vmin = self.vminic
                vmax = self.vmaxic
            elif (self.phase >= CL) | (self.phase <= DE):
                vmin = self.vminer
                vmax = self.vmaxer
            elif self.phase == AP:
                vmin = self.vminap
                vmax = self.vmaxap
            elif self.phase == GD:
                vmin = 0.0
                vmax = self.vmaxic

        # rotor
        elif self.lifttype == LIFT_ROTOR:
            vmin = self.vmin
            vmax = self.vmax
        return vmin, vmax

    def calc_axmax(self):
        # fix-wing
        if self.lifttype == LIFT_FIXWING:

            # in flight
            axmax = (self.max_thrust - self.drag) / self.mass

            # on ground
            if self.phase == GD:
                axmax = 2

        # drones
        elif self.lifttype == LIFT_ROTOR:
            axmax = 3.5

        # global minimum acceleration
        if axmax < 0.5:
            axmax = 0.5
        return axmax


class Plane:
    """This class provides a plane model based on the dynamics of the BlueSky simulator and the OpenAP performance model."""

    def __init__(self,role:str, dt: float, actype: str, lat: float, lon: float, alt: float, hdg: float, tas: float,plane_id:int) -> None:

        # setting
        self.n = None
        self.e = None
        self.s = None
        self.hist_len = None
        self.D_dest = None
        self.fly_to_goal = None
        self.s_old = None
        self.id = plane_id
        self.dt = dt
        self.actype = actype
        self.delta_hdg = 5.0  # [deg]
        self.delta_tas = 1.0  # [m/s]

        # positions
        self.lat = lat   # latitude [deg]
        self.lon = lon   # longitude [deg]
        self.alt = alt   # altitude [m]
        self.hdg = hdg   # heading [deg]
        self.trk = hdg   # track angle [deg]
        self.vs  = 0.0   # vertical speed [m/s]

        # velocities
        self.tas = tas                    # true airspeed [m/s]
        self.cas = aero.tas2cas(tas, alt) # calibrated airspeed [m/s]
        self.gs  = tas                    # ground speed [m/s]
        hdgrad   = np.radians(hdg)
        self.gsnorth = self.tas * np.cos(hdgrad) # ground speed [m/s]
        self.gseast = self.tas * np.sin(hdgrad)  # ground speed [m/s]

        # acceleration
        self.ax = 0.0  # [m/s²] current longitudinal acceleration

        # atmosphere
        self.p, self.rho, self.Temp = aero.vatmos(alt) 

        # miscallaneous
        self.coslat = np.cos(np.radians(lat))

        # behavior type
        assert role in ["RL", "VFG", "RND", "CUT"], "Unknown behavior type of the plane."
        self.role = role

    def upd_dynamics(self, perf:OpenAP, discrete_acts=False, a=None, dest=None):
        #---------- Atmosphere --------------------------------
        self.p, self.rho, self.Temp = aero.vatmos(self.alt)

        #---------- Fly the Aircraft --------------------------
        if self.role == "RL":
            self.cnt_hdg, self.cnt_tas = self._RL_control(a, discrete_acts)

        elif self.role == "VFG":
            self.cnt_hdg, self.cnt_tas = self._VFG_control()

        elif self.role == "RND":
            self.cnt_hdg, self.cnt_tas = self._RND_control()

        elif self.role == "CUT":
            self.cnt_hdg, self.cnt_tas = self._CUT_control()

        #---------- Performance Update ------------------------
        perf.update(self.tas, self.vs, self.alt, self.ax)
        self.axmax = perf.axmax

        #---------- Limit commanded speeds based on performance ------------------------------
        self.cnt_tas, self.vs, self.alt = perf.limits(self.cnt_tas, self.vs, self.alt, self.ax)

        #---------- Kinematics --------------------------------
        self._update_airspeed()
        self._update_groundspeed()
        self._update_pos(dest)

    def _RL_control(self, a, discrete_acts):
        """a is np.array(2,) containing delta tas and delta heading, or np.array(1,) containing only delta heading."""
        # store action for rendering
        self.action = a
        assert all([-1 <= ele <= 1 for ele in a]), "Actions need to be in [-1,1]."

        if len(a) == 2:
            return [self.hdg + self.delta_hdg * a[0], self.tas + self.delta_tas * a[1]]
        else:
            return [self.hdg + self.delta_hdg * a[0], self.tas]

    def _CUT_control(self):
        return self.hdg, self.tas

    def _VFG_control(self):
        return rtd(self.dc), self.tas

    def _RND_control(self):
        return self.hdg + float(np.random.uniform(low=-self.delta_hdg, high=self.delta_hdg)), self.tas

    def _update_airspeed(self):
        """Note: We perform no update of vertical speed since we stay at a specific altitude."""
        # compute horizontal acceleration
        delta_spd = self.cnt_tas - self.tas
        need_ax = np.abs(delta_spd) > np.abs(self.dt * self.axmax)
        self.ax = need_ax * np.sign(delta_spd) * self.axmax
        
        # update velocities
        if need_ax:
            self.tas += self.ax * self.dt
        else:
            self.tas = self.cnt_tas
        self.cas = float(aero.vtas2cas(self.tas, self.alt))

        # update heading
        self.hdg = self.cnt_hdg % 360

    def _update_groundspeed(self):
        """Compute ground speed and track from heading and airspeed. Currently, we assume there is no wind."""
        self.gsnorth = self.tas * np.cos(np.radians(self.hdg))
        self.gseast  = self.tas * np.sin(np.radians(self.hdg))
        self.gs  = self.tas
        self.trk = self.hdg
        self.windnorth, self.windeast = 0.0, 0.0

    def _update_pos(self, dest):
        """Euler update. Artficially keeps plane on a specific lat-lon area if a map is provided.
        Args:
            dest (Destination or None): contains lat-lon of center point and radius."""
        if dest is not None:
            self.alt_old = copy(self.alt)
            self.lat_old = copy(self.lat)
            self.coslat_old = copy(self.coslat)
            self.lon_old = copy(self.lon)

        self.alt = np.round(self.alt + self.vs * self.dt, 6)
        self.lat = self.lat + np.degrees(self.dt * self.gsnorth / aero.Rearth)
        self.coslat = np.cos(np.deg2rad(self.lat))
        self.lon = self.lon + np.degrees(self.dt * self.gseast / self.coslat / aero.Rearth)

        if dest is not None:
            if latlondist(latd1=self.lat, lond1=self.lon, latd2=dest.lat, lond2=dest.lon) >= dest.respawn_radius:
                self.alt = self.alt_old
                self.lat = self.lat_old
                self.coslat = self.coslat_old
                self.lon = self.lon_old

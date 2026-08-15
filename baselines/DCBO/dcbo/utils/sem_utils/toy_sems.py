from collections import OrderedDict
import numpy as np


class PISHCAT_SEM:
    @staticmethod
    def static():

        P = lambda noise, t, sample: noise
        I = lambda noise, t, sample: noise
        S = lambda noise, t, sample: sample["I"][t] + noise
        H = lambda noise, t, sample: sample["P"][t] + noise
        C = lambda noise, t, sample: sample["H"][t] + noise
        A = lambda noise, t, sample: sample["I"][t] + sample["P"][t] + noise
        T = lambda noise, t, sample: sample["C"][t] + sample["A"][t] + noise
        return OrderedDict([("P", P), ("I", I), ("S", S), ("H", H), ("C", C), ("A", A), ("T", T)])

    @staticmethod
    def dynamic():

        P = lambda noise, t, sample: sample["P"][t - 1] + noise
        I = lambda noise, t, sample: sample["I"][t - 1] + noise
        S = lambda noise, t, sample: sample["S"][t - 1] + sample["I"][t] + noise
        H = lambda noise, t, sample: sample["S"][t - 1] + sample["P"][t] + noise
        C = lambda noise, t, sample: sample["H"][t] + noise
        A = lambda noise, t, sample: sample["I"][t] + sample["P"][t] + noise
        T = lambda noise, t, sample: sample["C"][t] + sample["A"][t] + noise
        return OrderedDict([("P", P), ("I", I), ("S", S), ("H", H), ("C", C), ("A", A), ("T", T)])


class StationaryDependentSEM:
    @staticmethod
    def static():

        X = lambda noise, t, sample: noise
        Z = lambda noise, t, sample: np.exp(-sample["X"][t]) + noise
        Y = lambda noise, t, sample: np.cos(sample["Z"][t]) - np.exp(-sample["Z"][t] / 20.0) + noise
        return OrderedDict([("X", X), ("Z", Z), ("Y", Y)])

    @staticmethod
    def dynamic():

        # We get temporal innovation by introducing transfer functions between temporal indices
        X = lambda noise, t, sample: sample["X"][t - 1] + noise
        Z = lambda noise, t, sample: np.exp(-sample["X"][t]) + sample["Z"][t - 1] + noise
        Y = (
            lambda noise, t, sample: np.cos(sample["Z"][t])
            - np.exp(-sample["Z"][t] / 20.0)
            + sample["Y"][t - 1]
            + noise
        )
        return OrderedDict([("X", X), ("Z", Z), ("Y", Y)])


class LinearMultipleChildrenSEM:
    """
    Test DAG for nodes within a slice that have more than one child _within_ the slice.

    Returns
    -------
        None
    """

    @staticmethod
    def static() -> OrderedDict:

        X = lambda noise, t, sample: 1 + noise
        Z = lambda noise, t, sample: 2 * sample["X"][t] + noise
        Y = lambda noise, t, sample: 2 * sample["Z"][t] - sample["X"][t] + noise
        return OrderedDict([("X", X), ("Z", Z), ("Y", Y)])

    @staticmethod
    def dynamic() -> OrderedDict:

        # We get temporal innovation by introducing transfer functions between temporal indices
        X = lambda noise, t, sample: sample["X"][t - 1] + 1 + noise
        Z = lambda noise, t, sample: 2 * sample["X"][t] + sample["Z"][t - 1] + noise
        Y = lambda noise, t, sample: 2 * sample["Z"][t] + sample["Y"][t - 1] - sample["X"][t] + noise
        return OrderedDict([("X", X), ("Z", Z), ("Y", Y)])


class StationaryDependentMultipleChildrenSEM:
    """
    Test DAG for nodes within a slice that have more than one child _within_ the slice.

    Returns
    -------
        None
    """

    @staticmethod
    def static() -> OrderedDict:

        X = lambda noise, t, sample: noise
        Z = lambda noise, t, sample: np.exp(-sample["X"][t]) + noise
        Y = (
            lambda noise, t, sample: np.cos(sample["Z"][t])
            - np.exp(-sample["Z"][t] / 20.0)
            - np.sin(sample["X"][t])
            + noise
        )
        return OrderedDict([("X", X), ("Z", Z), ("Y", Y)])

    @staticmethod
    def dynamic() -> OrderedDict:

        # We get temporal innovation by introducing transfer functions between temporal indices
        X = lambda noise, t, sample: sample["X"][t - 1] + noise
        Z = lambda noise, t, sample: np.exp(-sample["X"][t]) + sample["Z"][t - 1] + noise
        Y = (
            lambda noise, t, sample: np.cos(sample["Z"][t])
            - np.exp(-sample["Z"][t] / 20.0)
            + sample["Y"][t - 1]
            - np.sin(sample["X"][t])
            + noise
        )
        return OrderedDict([("X", X), ("Z", Z), ("Y", Y)])


class StationaryIndependentSEM:
    @staticmethod
    def static():
        X = lambda noise, t, sample: noise
        Z = lambda noise, t, sample: noise
        Y = (
            lambda noise, t, sample: -2 * np.exp(-((sample["X"][t] - 1) ** 2) - (sample["Z"][t] - 1) ** 2)
            - np.exp(-((sample["X"][t] + 1) ** 2) - sample["Z"][t] ** 2)
            + noise
        )
        return OrderedDict([("X", X), ("Z", Z), ("Y", Y)])

    @staticmethod
    def dynamic():
        X = lambda noise, t, sample: -sample["X"][t - 1] + noise
        Z = lambda noise, t, sample: -sample["Z"][t - 1] + noise
        Y = (
            lambda noise, t, sample: -2 * np.exp(-((sample["X"][t] - 1) ** 2) - (sample["Z"][t] - 1) ** 2)
            - np.exp(-((sample["X"][t] + 1) ** 2) - sample["Z"][t] ** 2)
            + sample["Y"][t - 1]
            + noise
        )
        return OrderedDict([("X", X), ("Z", Z), ("Y", Y)])


class NonStationaryDependentSEM:
    """
    This SEM currently supports one change point.

    This SEM changes topology over t.

    with: intervention_domain = {'X':[-4,1],'Z':[-3,3]}
    """

    def __init__(self, change_point):
        """
        Initialise change point(s).

        Parameters
        ----------
        cp : int
            The temporal index of the change point (cp).
        """
        self.cp = change_point

    @staticmethod
    def static():
        """
        noise: e
        sample: s
        time index: t
        """
        X = lambda e, t, s: e
        Z = lambda e, t, s: s["X"][t] + e
        Y = lambda e, t, s: np.sqrt(abs(36 - (s["Z"][t] - 1) ** 2)) + 1 + e
        return OrderedDict([("X", X), ("Z", Z), ("Y", Y)])

    def dynamic(self):
        X = lambda e, t, s: s["X"][t - 1] + e
        Z = (
            lambda e, t, s: -s["X"][t] / s["X"][t - 1] + s["Z"][t - 1] + e
            if t == self.cp
            else s["X"][t] + s["Z"][t - 1] + e
        )
        Y = (
            lambda e, t, s: s["Z"][t] * np.cos(np.pi * s["Z"][t]) - s["Y"][t - 1] + e
            if t == self.cp
            else abs(s["Z"][t]) - s["Y"][t - 1] - s["Z"][t - 1] + e
        )
        return OrderedDict([("X", X), ("Z", Z), ("Y", Y)])


class NonStationaryIndependentSEM:
    """
    This SEM currently supports one change point.

    This SEM changes topology over t.
    """

    def __init__(self, change_point):
        self.change_point = change_point

    @staticmethod
    def static():
        X = lambda noise, t, sample: noise
        Z = lambda noise, t, sample: noise
        Y = (
            lambda noise, t, sample: -(
                2 * np.exp(-((sample["X"][t] - 1) ** 2) - (sample["Z"][t] - 1) ** 2)
                + np.exp(-((sample["X"][t] + 1) ** 2) - sample["Z"][t] ** 2)
            )
            + noise
        )
        return OrderedDict([("X", X), ("Z", Z), ("Y", Y)])


class CompleteGraphSEM:
    """
    SEM for the Synthetic (CompleteGraph) SCM with observed nodes F, A, B, C, D, E, Y
    and explicit latent confounders U1 (shared by A and Y) and U2 (shared by B and Y):
    A = F^2 + U1; B = U2; Y = cos(D) + sin(E) + U1 + U2*eps_y with eps_y ~ N(0, 0.1).
    U1/U2 are modelled as explicit exogenous nodes so the confounding structure
    is preserved (DCBO has no native latent support).
    """

    @staticmethod
    def static() -> OrderedDict:

        U1 = lambda noise, t, s: noise
        U2 = lambda noise, t, s: noise
        F = lambda noise, t, s: noise
        A = lambda noise, t, s: (s["F"][t] ** 2) + s["U1"][t]
        B = lambda noise, t, s: s["U2"][t]
        C = lambda noise, t, s: np.exp(-s["B"][t])
        D = lambda noise, t, s: np.exp(-s["C"][t]) / 10.0
        E = lambda noise, t, s: np.cos(s["A"][t]) + s["C"][t] / 10.0
        Y = lambda noise, t, s: np.cos(s["D"][t]) + np.sin(s["E"][t]) + s["U1"][t] + s["U2"][t] * (0.1 * noise)

        return OrderedDict([("U1", U1), ("U2", U2), ("F", F), ("A", A), ("B", B), ("C", C), ("D", D), ("E", E), ("Y", Y)])

    @staticmethod
    def dynamic() -> OrderedDict:
        # Simple carry-over dynamics; not used when T == 1
        U1 = lambda e, t, s: s["U1"][t - 1] + e
        U2 = lambda e, t, s: s["U2"][t - 1] + e
        F = lambda e, t, s: s["F"][t - 1] + e
        A = lambda e, t, s: (s["F"][t] ** 2) + s["U1"][t]
        B = lambda e, t, s: s["U2"][t] + s["B"][t - 1]
        C = lambda e, t, s: np.exp(-s["B"][t]) + e
        D = lambda e, t, s: np.exp(-s["C"][t]) / 10.0 + e
        E = lambda e, t, s: np.cos(s["A"][t]) + s["C"][t] / 10.0 + e
        Y = lambda e, t, s: np.cos(s["D"][t]) + np.sin(s["E"][t]) + s["U1"][t] + s["U2"][t] * (0.1 * e)

        return OrderedDict([("U1", U1), ("U2", U2), ("F", F), ("A", A), ("B", B), ("C", C), ("D", D), ("E", E), ("Y", Y)])


class PSA_SEM:
    """
    PSA healthcare SEM (static preferred; dynamic provides simple carry-over).

    Variables: Age, BMI, Aspirin, Statin, cancer, PSA
    Causal order: Age -> BMI -> Aspirin, Statin -> cancer -> PSA
    """

    @staticmethod
    def static() -> OrderedDict:
        # Age ~ Uniform(55, 75) -- we model as exogenous noise
        Age = lambda noise, t, s: noise  # supply noise ~ U(55,75) in sampling
        # BMI ~ Normal(27.0 - 0.01 * Age, 0.7)
        BMI = lambda noise, t, s: 27.0 - 0.01 * s["Age"][t] + noise
        # Aspirin ~ Sigmoid(-8.0 + 0.10 * Age + 0.03 * BMI) + noise
        Aspirin = lambda noise, t, s: 1.0 / (1.0 + np.exp(-(-8.0 + 0.10 * s["Age"][t] + 0.03 * s["BMI"][t]))) + noise
        # Statin ~ Sigmoid(-13.0 + 0.10 * Age + 0.20 * BMI) + noise
        Statin = lambda noise, t, s: 1.0 / (1.0 + np.exp(-(-13.0 + 0.10 * s["Age"][t] + 0.20 * s["BMI"][t]))) + noise
        # cancer ~ Sigmoid(2.2 - 0.05 * Age + 0.01 * BMI - 0.04 * Statin + 0.02 * Aspirin) + noise
        cancer = (
            lambda noise, t, s: 1.0
            / (1.0 + np.exp(-(
                2.2 - 0.05 * s["Age"][t] + 0.01 * s["BMI"][t] - 0.04 * s["Statin"][t] + 0.02 * s["Aspirin"][t]
            )))
            + noise
        )
        # PSA ~ Normal(6.8 + 0.04 * Age - 0.15 * BMI - 0.60 * Statin + 0.55 * Aspirin + 1.00 * cancer, 0.4)
        PSA = (
            lambda noise, t, s: 6.8
            + 0.04 * s["Age"][t]
            - 0.15 * s["BMI"][t]
            - 0.60 * s["Statin"][t]
            + 0.55 * s["Aspirin"][t]
            + 1.00 * s["cancer"][t]
            + noise
        )
        return OrderedDict([("Age", Age), ("BMI", BMI), ("Aspirin", Aspirin), ("Statin", Statin), ("cancer", cancer), ("PSA", PSA)])

    @staticmethod
    def dynamic() -> OrderedDict:
        Age = lambda e, t, s: s["Age"][t - 1] + e
        BMI = lambda e, t, s: 27.0 - 0.01 * s["Age"][t] + e
        Aspirin = lambda e, t, s: 1.0 / (1.0 + np.exp(-(-8.0 + 0.10 * s["Age"][t] + 0.03 * s["BMI"][t]))) + e
        Statin = lambda e, t, s: 1.0 / (1.0 + np.exp(-(-13.0 + 0.10 * s["Age"][t] + 0.20 * s["BMI"][t]))) + e
        cancer = (
            lambda e, t, s: 1.0
            / (1.0 + np.exp(-(
                2.2 - 0.05 * s["Age"][t] + 0.01 * s["BMI"][t] - 0.04 * s["Statin"][t] + 0.02 * s["Aspirin"][t]
            )))
            + e
        )
        PSA = (
            lambda e, t, s: 6.8
            + 0.04 * s["Age"][t]
            - 0.15 * s["BMI"][t]
            - 0.60 * s["Statin"][t]
            + 0.55 * s["Aspirin"][t]
            + 1.00 * s["cancer"][t]
            + e
        )
        return OrderedDict([("Age", Age), ("BMI", BMI), ("Aspirin", Aspirin), ("Statin", Statin), ("cancer", cancer), ("PSA", PSA)])

    def dynamic(self):
        #  X_t | X_{t-1}
        X = lambda noise, t, sample: sample["X"][t - 1] + noise
        Z = (
            lambda noise, t, sample: np.cos(sample["Z"][t - 1]) + noise
            if t == self.change_point
            else np.sin(sample["Z"][t - 1] ** 2) * sample["X"][t - 1] + noise
        )
        #  if t <= 1: Y_t | Z_t, Y_{t-1} else: Y_t | Z_t, X_t, Y_{t-1}
        Y = (
            lambda noise, t, sample:
            # np.exp(-np.cos(sample["X"][t]) ** 2)
            -np.exp(-(sample["Z"][t]) / 3.0)
            + np.exp(-sample["X"][t] / 3.0)
            + sample["Y"][t - 1]
            + sample["X"][t - 1]
            + noise
            if t == self.change_point
            else -2 * np.exp(-((sample["X"][t]) ** 2) - (sample["Z"][t] - sample["Z"][t - 1]) ** 2)
            - np.exp(-((sample["X"][t] - sample["Z"][t]) ** 2))
            + np.cos(sample["Z"][t])
            - sample["Y"][t - 1]
            + noise
        )
        return OrderedDict([("X", X), ("Z", Z), ("Y", Y)])


class PSA_CDC_SEM:
    """
    PSA-cdc SEM with linear relationships from provided equations.
    Exogenous: Age (sampled Uniform[40,80] at t=0; carry-over for t>0)
    Endogenous: Aspirin, Statin, BMI, cancer, PSA
    Topological order (within-slice): Age -> Aspirin -> Statin -> BMI -> cancer -> PSA
    """

    @staticmethod
    def static() -> OrderedDict:
        Age = lambda noise, t, s: np.random.uniform(40.0, 80.0)
        Aspirin = lambda noise, t, s: -0.11150676125977188 + 0.0028351700976795165 * s["Age"][t] + noise
        Statin = (
            lambda noise, t, s: -0.4301116250265201
            + 0.011962926923648507 * s["Age"][t]
            + 0.22879470732551127 * s["Aspirin"][t]
            + noise
        )
        BMI = (
            lambda noise, t, s: 31.909557845884837
            - 0.057206596823079305 * s["Age"][t]
            + 1.8024468168006789 * s["Statin"][t]
            + noise
        )
        cancer = lambda noise, t, s: -0.03219056548415463 + 0.000624055746190663 * s["Age"][t] + noise
        PSA = (
            lambda noise, t, s: -1.7212497864635612
            + 0.07057053193905057 * s["Age"][t]
            + 4.93154199589294 * s["cancer"][t]
            - 0.02146297604133429 * s["BMI"][t]
            + noise
        )
        return OrderedDict([
            ("Age", Age), ("Aspirin", Aspirin), ("Statin", Statin), ("BMI", BMI), ("cancer", cancer), ("PSA", PSA)
        ])

    @staticmethod
    def dynamic() -> OrderedDict:
        Age = lambda e, t, s: s["Age"][t - 1]
        Aspirin = lambda e, t, s: -0.11150676125977188 + 0.0028351700976795165 * s["Age"][t] + e
        Statin = (
            lambda e, t, s: -0.4301116250265201
            + 0.011962926923648507 * s["Age"][t]
            + 0.22879470732551127 * s["Aspirin"][t]
            + e
        )
        BMI = (
            lambda e, t, s: 31.909557845884837
            - 0.057206596823079305 * s["Age"][t]
            + 1.8024468168006789 * s["Statin"][t]
            + e
        )
        cancer = lambda e, t, s: -0.03219056548415463 + 0.000624055746190663 * s["Age"][t] + e
        PSA = (
            lambda e, t, s: -1.7212497864635612
            + 0.07057053193905057 * s["Age"][t]
            + 4.93154199589294 * s["cancer"][t]
            - 0.02146297604133429 * s["BMI"][t]
            + e
        )
        return OrderedDict([
            ("Age", Age), ("Aspirin", Aspirin), ("Statin", Statin), ("BMI", BMI), ("cancer", cancer), ("PSA", PSA)
        ])


class DIABETES_SEM:
    """
    Diabetes SEM per provided linear relations.
    Exogenous: DiabetesPedigreeFunction (DPF), Age
    Endogenous: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, Outcome
    """

    @staticmethod
    def static() -> OrderedDict:
        DiabetesPedigreeFunction = lambda e, t, s: np.random.uniform(0.078, 2.42)
        Age = lambda e, t, s: np.random.uniform(21.0, 81.0)
        Pregnancies = lambda e, t, s: -1.3394 + 0.16 * s["Age"][t] + e
        BloodPressure = lambda e, t, s: 56.2742 + 0.39 * s["Age"][t] + e
        SkinThickness = (
            lambda e, t, s: 10.6921
            + 0.20 * s["BloodPressure"][t]
            + 8.59 * s["DiabetesPedigreeFunction"][t]
            - 0.24 * s["Age"][t]
            + e
        )
        Insulin = lambda e, t, s: 3.2963 + 3.00 * s["SkinThickness"][t] + 31.48 * s["DiabetesPedigreeFunction"][t] + e
        BMI = lambda e, t, s: 23.2497 + 0.08 * s["BloodPressure"][t] + 0.17 * s["SkinThickness"][t] + e
        Glucose = (
            lambda e, t, s: 74.4170
            - 0.21 * s["SkinThickness"][t]
            + 0.10 * s["Insulin"][t]
            + 0.66 * s["BMI"][t]
            + 0.66 * s["Age"][t]
            + e
        )
        Outcome = (
            lambda e, t, s: 1.7908
            - 0.02 * s["Pregnancies"][t]
            - 0.01 * s["Glucose"][t]
            + 0.00 * s["BloodPressure"][t]
            - 0.01 * s["BMI"][t]
            - 0.10 * s["DiabetesPedigreeFunction"][t]
            + e
        )
        return OrderedDict([
            ("DiabetesPedigreeFunction", DiabetesPedigreeFunction),
            ("Age", Age),
            ("Pregnancies", Pregnancies),
            ("Glucose", Glucose),
            ("BloodPressure", BloodPressure),
            ("SkinThickness", SkinThickness),
            ("Insulin", Insulin),
            ("BMI", BMI),
            ("Outcome", Outcome),
        ])

    @staticmethod
    def dynamic() -> OrderedDict:
        DiabetesPedigreeFunction = lambda e, t, s: s["DiabetesPedigreeFunction"][t - 1]
        Age = lambda e, t, s: s["Age"][t - 1]
        Pregnancies = lambda e, t, s: -1.3394 + 0.16 * s["Age"][t] + e
        BloodPressure = lambda e, t, s: 56.2742 + 0.39 * s["Age"][t] + e
        SkinThickness = (
            lambda e, t, s: 10.6921
            + 0.20 * s["BloodPressure"][t]
            + 8.59 * s["DiabetesPedigreeFunction"][t]
            - 0.24 * s["Age"][t]
            + e
        )
        Insulin = lambda e, t, s: 3.2963 + 3.00 * s["SkinThickness"][t] + 31.48 * s["DiabetesPedigreeFunction"][t] + e
        BMI = lambda e, t, s: 23.2497 + 0.08 * s["BloodPressure"][t] + 0.17 * s["SkinThickness"][t] + e
        Glucose = (
            lambda e, t, s: 74.4170
            - 0.21 * s["SkinThickness"][t]
            + 0.10 * s["Insulin"][t]
            + 0.66 * s["BMI"][t]
            + 0.66 * s["Age"][t]
            + e
        )
        Outcome = (
            lambda e, t, s: 1.7908
            - 0.02 * s["Pregnancies"][t]
            - 0.01 * s["Glucose"][t]
            + 0.00 * s["BloodPressure"][t]
            - 0.01 * s["BMI"][t]
            - 0.10 * s["DiabetesPedigreeFunction"][t]
            + e
        )
        return OrderedDict([
            ("DiabetesPedigreeFunction", DiabetesPedigreeFunction),
            ("Age", Age),
            ("Pregnancies", Pregnancies),
            ("Glucose", Glucose),
            ("BloodPressure", BloodPressure),
            ("SkinThickness", SkinThickness),
            ("Insulin", Insulin),
            ("BMI", BMI),
            ("Outcome", Outcome),
        ])


class ChainSEM:
    """
    Chain SEM per chain.yaml:
    Z = -0.5 * X + U_Z
    Y = - W - 3 * Z * X + U_Y
    X and W exogenous.
    """

    @staticmethod
    def static() -> OrderedDict:
        X = lambda e, t, s: e
        W = lambda e, t, s: e
        Z = lambda e, t, s: -0.5 * s["X"][t] + e
        Y = lambda e, t, s: -s["W"][t] - 3.0 * s["Z"][t] * s["X"][t] + e
        return OrderedDict([("X", X), ("W", W), ("Z", Z), ("Y", Y)])

    @staticmethod
    def dynamic() -> OrderedDict:
        X = lambda e, t, s: s["X"][t - 1] + e
        W = lambda e, t, s: s["W"][t - 1] + e
        Z = lambda e, t, s: -0.5 * s["X"][t] + s["Z"][t - 1] + e
        Y = lambda e, t, s: -s["W"][t] - 3.0 * s["Z"][t] * s["X"][t] + s["Y"][t - 1] + e
        return OrderedDict([("X", X), ("W", W), ("Z", Z), ("Y", Y)])


class EcologySEM:
    """
    Ecology SEM (Bermuda reef calcification) matching sem/ecology_sem_equations.json.

    Exogenous: E (Tem), S (Sal), N (Nut), T (TA).
    Structure: X (pCO2) <- E; C (Chl-a) <- N; L (Light) <- C; P (pHsw) <- X;
               D (DIC) <- X; O (Omega_A) <- E, S, X; Y (NEC) <- L, N, P, O.
    Target: Y (maximize).  DCBO supplies standard-normal noise e; the means and
    standard deviations of the fitted SCM are embedded in the lambdas.
    """

    @staticmethod
    def static() -> OrderedDict:
        # Exogenous (Normal with fitted mean/std)
        E = lambda e, t, s: 24.184130434782606 + 3.2204054790158585 * e
        S = lambda e, t, s: 36.591623913043485 + 0.14919708693132147 * e
        N = lambda e, t, s: 0.4920652173913041 + 1.5924077981635778 * e
        T = lambda e, t, s: 2357.893695652174 + 27.609355287148222 * e
        # Endogenous (fitted linear mechanisms with residual-scaled noise)
        X = lambda e, t, s: 18.79817390034509 + 15.797383660309157 * s["E"][t] + 28.896638707243493 * e
        C = lambda e, t, s: 0.37342021355081695 - 0.0024002572713753005 * s["N"][t] + 0.039295137153226196 * e
        L = lambda e, t, s: 6665.081995591907 - 10737.462582329481 * s["C"][t] + 1546.9130344410867 * e
        P = lambda e, t, s: 8.427705924233592 - 0.0009655314986363629 * s["X"][t] + 0.005258257571036228 * e
        D = lambda e, t, s: 2131.6721072369096 - 0.2165604666734924 * s["X"][t] + 18.412099847093625 * e
        O = lambda e, t, s: (
            3.2452480541052733
            + 0.09433222790709161 * s["E"][t]
            + 0.006753920720864577 * s["S"][t]
            - 0.005737087151230129 * s["X"][t]
            + 0.03695797753398314 * e
        )
        Y = lambda e, t, s: (
            211.4225551648116
            - 3.012576882356712e-05 * s["L"][t]
            + 0.01667959146973974 * s["N"][t]
            - 25.71927722049691 * s["P"][t]
            - 0.5004034826468742 * s["O"][t]
            + 1.0733401618242278 * e
        )

        return OrderedDict([("E", E), ("S", S), ("N", N), ("T", T), ("X", X), ("C", C), ("L", L), ("P", P), ("D", D), ("O", O), ("Y", Y)])

    @staticmethod
    def dynamic() -> OrderedDict:
        # Temporal carry-over + same structural equations
        E = lambda e, t, s: 24.184130434782606 + 3.2204054790158585 * e + s["E"][t - 1]
        S = lambda e, t, s: 36.591623913043485 + 0.14919708693132147 * e + s["S"][t - 1]
        N = lambda e, t, s: 0.4920652173913041 + 1.5924077981635778 * e + s["N"][t - 1]
        T = lambda e, t, s: 2357.893695652174 + 27.609355287148222 * e + s["T"][t - 1]
        X = lambda e, t, s: 18.79817390034509 + 15.797383660309157 * s["E"][t] + 28.896638707243493 * e + s["X"][t - 1]
        C = lambda e, t, s: 0.37342021355081695 - 0.0024002572713753005 * s["N"][t] + 0.039295137153226196 * e + s["C"][t - 1]
        L = lambda e, t, s: 6665.081995591907 - 10737.462582329481 * s["C"][t] + 1546.9130344410867 * e + s["L"][t - 1]
        P = lambda e, t, s: 8.427705924233592 - 0.0009655314986363629 * s["X"][t] + 0.005258257571036228 * e + s["P"][t - 1]
        D = lambda e, t, s: 2131.6721072369096 - 0.2165604666734924 * s["X"][t] + 18.412099847093625 * e + s["D"][t - 1]
        O = lambda e, t, s: (
            3.2452480541052733
            + 0.09433222790709161 * s["E"][t]
            + 0.006753920720864577 * s["S"][t]
            - 0.005737087151230129 * s["X"][t]
            + 0.03695797753398314 * e
            + s["O"][t - 1]
        )
        Y = lambda e, t, s: (
            211.4225551648116
            - 3.012576882356712e-05 * s["L"][t]
            + 0.01667959146973974 * s["N"][t]
            - 25.71927722049691 * s["P"][t]
            - 0.5004034826468742 * s["O"][t]
            + 1.0733401618242278 * e
            + s["Y"][t - 1]
        )

        return OrderedDict([("E", E), ("S", S), ("N", N), ("T", T), ("X", X), ("C", C), ("L", L), ("P", P), ("D", D), ("O", O), ("Y", Y)])


class EpidemiologySEM:
    """
    Epidemiology SEM matching sem/epidemiology_sem_equations.json:
    B ~ U(-1, 1), T ~ U(4, 8) (exogenous)
    L = exp(0.5 * T + U) where U ~ N(0,1)
    R = 4 + L * T
    Y = 0.5 + cos(4*T) + sin(-L + 2*R) + B + eps where eps ~ N(0,1)
    Target: Y (minimize); interventions on L and B.
    """

    @staticmethod
    def static() -> OrderedDict:
        # Exogenous uniform draws (the supplied standard-normal e is unused)
        B = lambda e, t, s: np.random.uniform(-1.0, 1.0)
        T = lambda e, t, s: np.random.uniform(4.0, 8.0)
        # L depends on T with exponential relationship: L = exp(0.5 * T + U)
        L = lambda e, t, s: np.exp(0.5 * s["T"][t] + e)
        # R depends on L and T with interaction: R = 4 + L * T
        R = lambda e, t, s: 4.0 + s["L"][t] * s["T"][t]
        # Y = 0.5 + cos(4*T) + sin(-L + 2*R) + B + eps
        Y = lambda e, t, s: 0.5 + np.cos(4.0 * s["T"][t]) + np.sin(-s["L"][t] + 2.0 * s["R"][t]) + s["B"][t] + e

        return OrderedDict([("B", B), ("T", T), ("L", L), ("R", R), ("Y", Y)])

    @staticmethod
    def dynamic() -> OrderedDict:
        # Temporal carry-over + same structural equations
        B = lambda e, t, s: s["B"][t - 1] + e
        T = lambda e, t, s: s["T"][t - 1] + e
        L = lambda e, t, s: np.exp(0.5 * s["T"][t] + e) + s["L"][t - 1]
        R = lambda e, t, s: 4.0 + s["L"][t] * s["T"][t] + s["R"][t - 1]
        Y = lambda e, t, s: 0.5 + np.cos(4.0 * s["T"][t]) + np.sin(-s["L"][t] + 2.0 * s["R"][t]) + s["B"][t] + s["Y"][t - 1] + e

        return OrderedDict([("B", B), ("T", T), ("L", L), ("R", R), ("Y", Y)])


class ProteinSEM:
    """Linear-Gaussian SCM mirroring CBO_Benchmark/sem/protein_sem_equations.json.

    Causal order: PKC -> PKA -> {Raf, Mek, P38, Jnk, Akt} -> Erk
    PKC is exogenous. All endogenous nodes are linear regressions on their
    parents with additive Gaussian noise (the noise enters via the `e`
    argument; DCBO scales it externally).

    Target: Erk (minimize).
    """

    @staticmethod
    def static() -> OrderedDict:
        # Exogenous root
        PKC = lambda e, t, s: e
        PKA = lambda e, t, s: 554.3907305751273 + 0.8411525573301047 * s["PKC"][t] + e
        Raf = lambda e, t, s: (
            62.199045961323414
            - 0.17774509086121948 * s["PKC"][t]
            - 0.0003792075216956362 * s["PKA"][t]
            + e
        )
        Mek = lambda e, t, s: (
            -1.0902746220577555
            + 0.03966245723060245 * s["PKC"][t]
            - 0.0006520266851554879 * s["PKA"][t]
            + 0.5208447051064984 * s["Raf"][t]
            + e
        )
        P38 = lambda e, t, s: (
            15.144327731160235
            + 1.2347832903144826 * s["PKC"][t]
            + 0.000590540235751414 * s["PKA"][t]
            + e
        )
        Jnk = lambda e, t, s: (
            52.9536028509298
            - 0.7648011883501407 * s["PKC"][t]
            - 0.005306096821872689 * s["PKA"][t]
            + e
        )
        Akt = lambda e, t, s: -31.110747106386974 + 0.12890532877586242 * s["PKA"][t] + e
        Erk = lambda e, t, s: (
            -23.248743333459377
            + 0.08170673252473967 * s["PKA"][t]
            - 0.029885814321525 * s["Mek"][t]
            + e
        )
        return OrderedDict(
            [
                ("PKC", PKC),
                ("PKA", PKA),
                ("Raf", Raf),
                ("Mek", Mek),
                ("P38", P38),
                ("Jnk", Jnk),
                ("Akt", Akt),
                ("Erk", Erk),
            ]
        )

    @staticmethod
    def dynamic() -> OrderedDict:
        # Temporal carry-over + same structural equations
        PKC = lambda e, t, s: s["PKC"][t - 1] + e
        PKA = lambda e, t, s: 554.3907305751273 + 0.8411525573301047 * s["PKC"][t] + s["PKA"][t - 1] + e
        Raf = lambda e, t, s: (
            62.199045961323414
            - 0.17774509086121948 * s["PKC"][t]
            - 0.0003792075216956362 * s["PKA"][t]
            + s["Raf"][t - 1]
            + e
        )
        Mek = lambda e, t, s: (
            -1.0902746220577555
            + 0.03966245723060245 * s["PKC"][t]
            - 0.0006520266851554879 * s["PKA"][t]
            + 0.5208447051064984 * s["Raf"][t]
            + s["Mek"][t - 1]
            + e
        )
        P38 = lambda e, t, s: (
            15.144327731160235
            + 1.2347832903144826 * s["PKC"][t]
            + 0.000590540235751414 * s["PKA"][t]
            + s["P38"][t - 1]
            + e
        )
        Jnk = lambda e, t, s: (
            52.9536028509298
            - 0.7648011883501407 * s["PKC"][t]
            - 0.005306096821872689 * s["PKA"][t]
            + s["Jnk"][t - 1]
            + e
        )
        Akt = lambda e, t, s: (
            -31.110747106386974
            + 0.12890532877586242 * s["PKA"][t]
            + s["Akt"][t - 1]
            + e
        )
        Erk = lambda e, t, s: (
            -23.248743333459377
            + 0.08170673252473967 * s["PKA"][t]
            - 0.029885814321525 * s["Mek"][t]
            + s["Erk"][t - 1]
            + e
        )
        return OrderedDict(
            [
                ("PKC", PKC),
                ("PKA", PKA),
                ("Raf", Raf),
                ("Mek", Mek),
                ("P38", P38),
                ("Jnk", Jnk),
                ("Akt", Akt),
                ("Erk", Erk),
            ]
        )

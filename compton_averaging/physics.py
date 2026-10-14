"""Инициализация физического калькулятора спектра (snlcs)."""
import os

import numpy as np
import yaml


def init_spectrum_sqinrp(base_dir, a0, T, gamma0):
    # Ленивая загрузка snlcs-зависимостей: повторяем схему из snlcs/src/test.ipynb
    try:
        from snlcs.external_fields import ExternalField
        from snlcs.ebeam import eGaussBeam
        from snlcs.solvers import ParticleMotion, Spectrum
        from snlcs.methods.motion import MotionIntegrationMethod
        from snlcs.methods.spectrum import SpectrumMethod
    except Exception as e:
        raise RuntimeError(
            "Не удалось импортировать snlcs. Убедитесь, что пакет установлен "
            "(например, pip install -e snlcs/Spectrum_of_nonlinear_Compton_scattering)."
        ) from e

    path_other_params = os.path.join(base_dir, 'config/OtherParams.yaml')
    path_extf_params = os.path.join(base_dir, 'config/ExtFieldsParams.yaml')
    path_ebeam_params = os.path.join(base_dir, 'config/eBeamParams.yaml')

    extfields_params = ExternalField.from_yaml(path_extf_params)
    ebeam_params = eGaussBeam.from_yaml(path_ebeam_params)
    with open(path_other_params, 'r', encoding='utf-8') as file:
        other_params = yaml.safe_load(file)  

    np.random.seed(int(other_params["random_seed"]))

    eta = 1        
    z0 = 0 
    extfields_params.a0 = a0
    extfields_params.T = T
    extfields_params.eta = eta

    arr_t = np.linspace(-10*T, 10*T, 10000)
    in_state = np.array([0, 0, z0, 0, 0, -np.sqrt(gamma0**2 - 1)])
    in_states = np.array([in_state])

    spectrum_sqinrp = Spectrum(
        ext_field=extfields_params, 
        initial_states=in_states,
        arr_t=arr_t, 
        solver_method=MotionIntegrationMethod.BORIS, 
        spec_calc_method=SpectrumMethod.QUAD_INTERP,
        coherent=False,
    )
    return spectrum_sqinrp

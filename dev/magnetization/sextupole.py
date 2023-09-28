import torch
from bmadx.bmad_torch.track_torch import Beam

def track_sextupole(beam, L, k2):
    x = beam.x
    px = beam.px
    y = beam.y
    py = beam.py
    z = beam.z
    pz = beam.pz

    xp = px / beam.p0c
    yp = py / beam.p0c

    x_f = x + L*xp - k2 * (
            (L ** 2 / 4)*(x**2 - y**2) +
            (L ** 3 / 12)*(x * xp - y * yp) +
            (L ** 4 / 24)*(xp**2 - yp**2)
    )
    xp_f = xp - k2 * (
            (L / 2)*(x**2 - y**2) +
            (L ** 2 / 4)*(x * xp - y * yp) +
            (L ** 3 / 6)*(xp**2 - yp**2)
    )

    y_f = y + L*yp + k2 * (
            (L ** 2 / 4) * (x * y) +
            (L ** 3 / 12) * (x * yp + y * xp) +
            (L ** 4 / 24) * (xp * yp)
    )
    yp_f = yp + k2 * (
            (L / 2) * (x * y) +
            (L ** 2 / 4) * (x * yp + y * xp) +
            (L ** 3 / 6) * (xp * yp)
    )

    # NOTE: this ignores chromatic focusing effects
    f_data = torch.cat(
        (
            x_f, xp_f * beam.p0c, y_f, yp_f * beam.p0c, beam.z, beam.pz
        )
    )

    return Beam(f_data, beam.p0c)






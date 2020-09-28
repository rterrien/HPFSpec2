
# HPFspec2

A versatile hpf spectrum class. Extends and modifies Gummi's original HPFSpec.

Capabilities:

- Easily calculate barycentric correction and BJD times
- Calculate CCFs for different orders
- Calculate absolute RVs for different orders using CCFs
- Calculate vsinis using a CCF method (uses a slowly rotating calibration star)
- Load and manipulate model spectra
- Create, modify, and test stellar masks for CCF calculation
- Fit line or CCF profiles
- Resample spectra with a variety of methods
- Combine spectra
- Translate spectrum into SpecUtils spectrum for additional functionality.

# Dependencies
Depends on the CCF module (in use by Gummi and Ryan), which has some fortran extensions for speed.

# Todo
Clean up and clarify CCF dependencies.
Add feature measurement methods.

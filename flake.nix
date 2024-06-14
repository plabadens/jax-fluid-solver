{
  description = "JAX fluid solver";

  inputs = {
    flake-utils.url = "github:numtide/flake-utils";
    flake-compat.url = "github:edolstra/flake-compat";
  };

  outputs = { self, nixpkgs, flake-utils, flake-compat }:
    flake-utils.lib.eachDefaultSystem (system:
      let 
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
        };
        pythonWithPkgs = pkgs.python311.withPackages (p: with p; [
          pip
          virtualenv
          wheel
        ]);
        cudaPkgs = pkgs.cudaPackages_12;
        FHSWithCuda = pkgs.buildFHSUserEnv {
          name = "cuda-env";
          targetPkgs = p: (with p; [
            pythonWithPkgs

            stdenv.cc
            zlib
          ]);
          profile = ''
            export EXTRA_LDFLAGS="-L/lib -L/run/opengl-driver/lib"
            export EXTRA_CCFLAGS="-I/usr/include"

            ACTIVATE="./venv/bin/activate"

            if [ -f $ACTIVATE ]; then
              source $ACTIVATE
            fi

            export JAX_SKIP_CUDA_CONSTRAINTS_CHECK=1
          '';
        };
      in
      {
        packages = rec {
          cuda = FHSWithCuda;
          default = cuda;
        };
        devShells = rec {
          cuda = FHSWithCuda.env;
          default = cuda;
        };
      }
    );
}

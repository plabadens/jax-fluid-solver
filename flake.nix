{
  description = "JAX fluid solver";

  inputs.flake-utils.url = "github:numtide/flake-utils";

  outputs = { self, nixpkgs, flake-utils }:
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
        cudaPkgs = pkgs.cudaPackages_11;
        FHSWithCuda = pkgs.buildFHSUserEnv {
          name = "cuda-env";
          targetPkgs = p: (with p; [
            cudaPkgs.cudatoolkit
            cudaPkgs.cudnn
            stdenv.cc
            zlib
          ]);
          profile = ''
            export CUDA_PATH=${cudaPkgs.cudatoolkit}

            export EXTRA_LDFLAGS="-L/lib -L/run/opengl-driver/lib"
            export EXTRA_CCFLAGS="-I/usr/include"

            ACTIVATE="./venv/bin/activate"

            if [ -f $ACTIVATE ]; then
              source $ACTIVATE
            fi
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

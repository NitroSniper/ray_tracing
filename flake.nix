{
  description = "A flake for developing and building rust application to binary";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

    flake-utils.url = "github:numtide/flake-utils";

    # rust dev toolchain
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    # cache build steps
    crane = {
      url = "github:ipetkov/crane";
    };
  };

  outputs =
    inputs:
    inputs.flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import inputs.nixpkgs {
          inherit system;
          overlays = [ (import inputs.rust-overlay) ];
        };

        utilities = with pkgs; [
          rust-analyzer
          bacon
        ];

        # Tell Crane to use our toolchain
        craneLib = (inputs.crane.mkLib pkgs).overrideToolchain (
          p: p.rust-bin.nightly.latest.default.override { }
        );

        # Common arguments can be set here to avoid repeating them later
        commonArgs = {
          src = craneLib.cleanCargoSource ./.;
          strictDeps = true;

          # Additional environment variables can be set directly
          # MY_CUSTOM_VAR = "some value";
        };

        # Build *just* the cargo dependencies, so we can reuse
        # all of that work (e.g. via cachix) when running in CI

        cargoArtifacts = craneLib.buildDepsOnly commonArgs;
        bin = craneLib.buildPackage (commonArgs // { inherit cargoArtifacts; });

        docker = pkgs.dockerTools.buildImage {
          name = "nox";
          # tag = "latest";
          # TODO! replace foo with package name
          config.Cmd = [ "${bin}/bin/foo" ];
        };
      in
      {
        packages = {
          # Build the binary itself, reusing the dependency
          # artifacts from above.
          inherit docker;
          default = bin;
        };
        devShells =
          let
            util = pkgs.mkShell { packages = utilities; };
            battery = craneLib.devShell { packages = utilities; };
            chain = craneLib.devShell { };
          in
          {
            inherit battery chain util;
            default = battery;
          };

      }
    );
}

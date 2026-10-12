{
  description = "Development environment for r2spec";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/8ce4ef6cb6f871616146b9fe26d2a5ae594e94fe";

  outputs = { nixpkgs, ... }:
    let
      system = "aarch64-linux";
      pkgs = import nixpkgs { inherit system; };
    in
    {
      devShells.${system}.default = pkgs.mkShell {
        packages = [
          pkgs.radare2
          (pkgs.python3.withPackages (ps: [
            ps.r2pipe
            ps.pytest
          ]))
          pkgs.rustc
          pkgs.cargo
          pkgs.bubblewrap
        ];
      };
    };
}

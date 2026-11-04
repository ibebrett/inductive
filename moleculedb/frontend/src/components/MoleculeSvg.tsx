import { useEffect, useState } from 'react'
import { getRDKit } from '../rdkit'

/** Draw a SMILES string (molecules or fragments like "[*:1]Cl") with RDKit.js. */
export function MoleculeSvg({ smiles, width = 250, height = 200 }: { smiles: string; width?: number; height?: number }) {
  const [svg, setSvg] = useState<string>()

  useEffect(() => {
    let cancelled = false
    getRDKit().then((rdkit) => {
      const mol = rdkit.get_mol(smiles)
      if (cancelled) return mol?.delete()
      setSvg(mol ? mol.get_svg(width, height) : undefined)
      mol?.delete()
    })
    return () => {
      cancelled = true
    }
  }, [smiles, width, height])

  return svg ? <div className="molecule" dangerouslySetInnerHTML={{ __html: svg }} /> : <div style={{ width, height }} />
}

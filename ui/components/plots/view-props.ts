import type {
  NormalizationMode,
  RegionPreset,
  StagedSpectrum,
} from "@/lib/types";

export interface ViewProps {
  staged: StagedSpectrum[];
  normalization: NormalizationMode;
  region: RegionPreset;
  linkXZoom: boolean;
  shareYScale: boolean;
  referenceFileId?: string;
}

from pathlib import Path


import typer
from datasets import load_dataset, load_from_disk
from torch.utils.data import Dataset


DATASET_ID = "sreesharvesh/transactiq-enriched"
SAVED_NAME = "transactiq_enriched_hf"




class MyDataset(Dataset):
   def __init__(self, data_path: Path) -> None:
       self.data_path = data_path
       self.data = None


   def load(self) -> None:
       obj = load_from_disk(str(self.data_path))
       if hasattr(obj, "keys"):
           split = "train" if "train" in obj.keys() else next(iter(obj.keys()))
           self.data = obj[split]
       else:
           self.data = obj


   def __len__(self) -> int:
       if self.data is None:
           raise RuntimeError("Dataset not loaded. Call load() first.")
       return self.data.num_rows


   def __getitem__(self, index: int):
       if self.data is None:
           raise RuntimeError("Dataset not loaded. Call load() first.")
       return self.data[index]


   def preprocess(self, output_folder: Path) -> None:
       output_folder.mkdir(parents=True, exist_ok=True)


       ds = load_dataset(DATASET_ID)


       split = "train" if "train" in ds.keys() else next(iter(ds.keys()))
       d = ds[split]


       print("splits:", list(ds.keys()))
       print("rows:", d.num_rows)
       print("columns:", d.column_names)
       print("first row:", d[0])


       n = min(5, d.num_rows)
       try:
           print(d.select(range(n)).to_pandas())
       except Exception:
           print(d.select(range(n)))


       ds.save_to_disk(str(output_folder / SAVED_NAME))




def preprocess(output_folder: Path = Path("data/raw")) -> None:
   print("Preprocessing data...")
   dataset = MyDataset(output_folder / SAVED_NAME)
   dataset.preprocess(output_folder)




if __name__ == "__main__":
   typer.run(preprocess)



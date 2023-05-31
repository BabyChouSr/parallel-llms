class DatasetCreator:
    def __init__(self, prompter, tokenizer, cutoff_len, train_on_inputs, add_eos_token, dataset_size, val_set_size):
        self.prompter = prompter
        self.tokenizer = tokenizer
        self.cutoff_len = cutoff_len
        self.train_on_inputs = train_on_inputs
        self.add_eos_token = add_eos_token
        self.val_set_size = val_set_size
        self.dataset_size = dataset_size

    def tokenize(self, prompt, add_eos_token=True):
            self.tokenizer.pad_token_id = (
                        0  # unk. we want this to be different from the eos token
                )
            self.tokenizer.padding_side = "left"
            # there's probably a way to do this with the tokenizer settings
            # but again, gotta move fast
            result = self.tokenizer(
                prompt,
                truncation=True,
                max_length=self.cutoff_len,
                padding=False,
                return_tensors=None,
            )
            if (
                result["input_ids"][-1] != self.tokenizer.eos_token_id
                and len(result["input_ids"]) < self.cutoff_len
                and add_eos_token
            ):
                result["input_ids"].append(self.tokenizer.eos_token_id)
                result["attention_mask"].append(1)

            result["labels"] = result["input_ids"].copy()

            return result

    def generate_and_tokenize_prompt(self, data_point):
            full_prompt = self.prompter.generate_prompt(
                data_point["instruction"],
                data_point["input"],
                data_point["output"],
            )
            tokenized_full_prompt = self.tokenize(full_prompt)
            if not self.train_on_inputs:
                user_prompt = self.prompter.generate_prompt(
                    data_point["instruction"], data_point["input"]
                )
                tokenized_user_prompt = self.tokenize(
                    user_prompt, add_eos_token=self.add_eos_token
                )
                user_prompt_len = len(tokenized_user_prompt["input_ids"])

                if self.add_eos_token:
                    user_prompt_len -= 1

                tokenized_full_prompt["labels"] = [
                    -100
                ] * user_prompt_len + tokenized_full_prompt["labels"][
                    user_prompt_len:
                ]  # could be sped up, probably
            return tokenized_full_prompt
    
    def create_dataset(self, data):        
        if self.val_set_size > 0:
            train_val = data["train"].train_test_split(
                test_size=self.val_set_size, shuffle=True, seed=42
            )
            train_data = (
                train_val["train"].shuffle().map(self.generate_and_tokenize_prompt)
            )
            val_data = (
                train_val["test"].shuffle().map(self.generate_and_tokenize_prompt)
            )
        else:
            train_data = data["train"].shuffle().map(self.generate_and_tokenize_prompt)

        return {
            "train": train_data,
            "val": val_data,
        }

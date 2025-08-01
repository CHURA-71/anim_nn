from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

class GPT2:
    def __init__(self, top_k=6):
        """
        GPT-2のトークン生成クラス
        
        Args:
            top_k (int): 上位何個のトークンを考慮するか（デフォルト: 6）
        """
        # GPT-2のトークナイザーとモデルをロード
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.model = GPT2LMHeadModel.from_pretrained("gpt2")
        
        # パディングトークンをeos_tokenに設定
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.top_k = top_k
    
    def __call__(self, input_tokens):
        """
        入力トークンから次のトークンを生成
        
        Args:
            input_tokens (list): 入力トークンのリスト
        
        Returns:
            tuple: (selected_token, token_list, selected_index)
                - selected_token: 選択されたトークン
                - token_list: 上位k個のトークンと確率の2次元リスト [[token, prob], ...]
                - selected_index: 選択されたトークンのインデックス (1-6)
        """
        # 入力が文字列ならエンコーディング
        if isinstance(input_tokens, str):
            input_tokens = self.tokenizer.encode(input_tokens, return_tensors='pt').tolist()[0]

        # リストをテンソルに変換
        if isinstance(input_tokens, list):
            inputs = torch.tensor([input_tokens])
        else:
            inputs = input_tokens
        
        # attention_maskを作成
        attention_mask = torch.ones(inputs.shape, dtype=torch.long)
        
        # 次のトークンを予測
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=inputs.shape[1] + 1,
                num_return_sequences=1,
                output_scores=True,
                return_dict_in_generate=True,
                attention_mask=attention_mask,
                do_sample=False,  # 確率的サンプリングを無効化
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # 次に来るトークンのスコアを取得
        logits = outputs.scores[-1]
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        
        # 上位k個のトークンを選択
        top_k_probs, top_k_indices = torch.topk(probabilities, self.top_k)
        
        # 上位k個のトークンをリストとして保持
        token_list = []
        for i in range(self.top_k):
            token = self.tokenizer.decode([top_k_indices[0][i].item()])
            prob = top_k_probs[0][i].item()
            token_list.append([token, prob])
        
        # 上位k個の確率から重み付きランダムサンプリング
        sampled_index = torch.multinomial(top_k_probs[0], 1).item()
        selected_token = top_k_indices[0][sampled_index].item()

        return selected_token, token_list, sampled_index + 1
    
    def decode_tokens(self, tokens):
        """
        トークンリストをテキストに変換
        
        Args:
            tokens (list): トークンのリスト
        
        Returns:
            str: デコードされたテキスト
        """
        return self.tokenizer.decode(tokens, skip_special_tokens=True)
    
    def encode_text(self, text):
        """
        テキストをトークンリストに変換
        
        Args:
            text (str): 入力テキスト
        
        Returns:
            list: トークンのリスト
        """
        return self.tokenizer.encode(text)


# 使用例
if __name__ == "__main__":
    # GPT2インスタンスを作成
    gpt2 = GPT2(top_k=6)
    
    # 初期テキスト
    input_text = "Once upon a time"
    input_tokens = gpt2.encode_text(input_text)
    
    print(f"Initial text: '{input_text}'")
    print(f"Initial tokens: {input_tokens}")
    print()

    max_length = 20  # 任意の長さを設定
    
    # 3回トークンを生成
    for step in range(max_length):
        token, token_list, token_idx = gpt2(input_tokens)
        
        print(f"Step {step + 1}:")
        print(f"Current text: '{gpt2.decode_tokens(input_tokens)}'")
        print("Top 6 tokens:")
        for i, (tok, prob) in enumerate(token_list):
            print(f"  {i}: ('{tok}', {prob*100:.1f}%)")
        print(f"Selected: Index {token_idx}, Token {token} ('{gpt2.tokenizer.decode([token])}')")
        print()
        
        # 選択されたトークンを追加
        input_tokens.append(token)
    
    print(f"Final generated text: '{gpt2.decode_tokens(input_tokens)}'")

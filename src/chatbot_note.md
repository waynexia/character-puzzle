# notes

- 每个句子的最大长度为10个词
- 包含了停用词的句子整个句子对都删除
- 编码gru采用双向rnn
- pack/pad 函数不需要对句子进行排序
- 双向gru要两个方向累加：
    ```python
    outputs = outputs[:, :, :self.hidden_size] + outputs[:, : ,self.hidden_size:]
    ```
- `For the decoder, we will manually feed our batch one time step at a time`
- 在计算loss的时候要加mask，和分类的时候取了时间序列的最后一个不同
- 两个训练时帮助收敛的技巧：`teacher forcing` & `gradient clipping`
- 训练时decoder要初始化
- 手动把timeseq一个个传给decoder




# stumb
- 我在处理数据的时候没有用mask，带来的影响有？
- 为什么decoder在embedding的时候要dropout

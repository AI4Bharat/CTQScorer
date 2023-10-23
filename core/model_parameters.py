class model_parameters():
    
    def __init__(self, name, algo='Greedy', use_8_bit=False, no_of_shots=4, 
    training_source='flores', testing_source='flores', src_lang='hin_Deva', dst_lang='eng_Latn',
    has_reranking=False, strategy='random_selection', strategy_nested='',
    inc_reranking=False, seed=10, diversify_prompts=False):
        self._name = name
        self._type_of_algo = algo
        self._use_8_bit = use_8_bit
        self._no_of_shots = no_of_shots
        self._training_source = training_source
        self._testing_source = testing_source
        self._src_lang = src_lang
        self._dst_lang = dst_lang
        self._has_reranking = has_reranking
        self._strategy = strategy
        self._strategy_nested = strategy_nested
        self._inc_reranking = inc_reranking
        self._seed = seed # planning to use these seeds: 10, 19, 42, 107, 141
        self._diversify_prompts = diversify_prompts

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, n):
        self._name = n

    @property
    def type_of_algo(self):
        return self._type_of_algo

    @type_of_algo.setter
    def type_of_algo(self, algo):
        self._type_of_algo = algo

    @property
    def use_8_bit(self):
        return self._use_8_bit

    @use_8_bit.setter
    def use_8_bit(self, flag):
        self._use_8_bit = flag

    @property
    def no_of_shots(self):
        return self._no_of_shots

    @no_of_shots.setter
    def no_of_shots(self, val):
        self._no_of_shots = val

    @property
    def training_source(self):
        return self._training_source

    @training_source.setter
    def training_source(self, val):
        self._training_source = val

    @property
    def testing_source(self):
        return self._testing_source

    @testing_source.setter
    def testing_source(self, val):
        self._testing_source = val

    @property
    def src_lang(self):
        return self._src_lang

    @src_lang.setter
    def src_lang(self, val):
        self._src_lang = val

    @property
    def dst_lang(self):
        return self._dst_lang

    @dst_lang.setter
    def dst_lang(self, val):
        self._dst_lang = val

    @property
    def has_reranking(self):
        return self._has_reranking

    @has_reranking.setter
    def has_reranking(self, val):
        self._has_reranking = val

    @property
    def strategy(self):
        return self._strategy

    @strategy.setter
    def strategy(self, val):
        self._strategy = val

    @property
    def strategy_nested(self):
        return self._strategy_nested

    @strategy_nested.setter
    def strategy_nested(self, val):
        self._strategy_nested = val

    @property
    def inc_reranking(self):
        return self._inc_reranking

    @inc_reranking.setter
    def inc_reranking(self, flag):
        self._inc_reranking = flag

    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, val):
        self._seed = val

    @property
    def diversify_prompts(self):
        return self._diversify_prompts

    @diversify_prompts.setter
    def diversify_prompts(self, flag):
        self._diversify_prompts = flag

    def __repr__(self):
        return 'name: {}, algo: {}, use_8_bit: {}, no_of_shots: {}, training_source: {}, testing_source: {}, src_lang: {}, dst_lang: {}, has_reranking: {}, strategy: {},  strategy_nested: {}, inc_reranking: {}, seed: {}, diversify_prompts: {}\n'.format(
            self.name, self.type_of_algo, self.use_8_bit, self.no_of_shots, self.training_source, self.testing_source, self.src_lang, self.dst_lang, self.has_reranking, self.strategy, self.strategy_nested, self.inc_reranking, self.seed, self.diversify_prompts)
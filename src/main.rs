use std::io::Write;

fn main() -> Result<(), String> {
    test()
}

fn test() -> Result<(), String> {
    'loop_start: loop {
        let mut line = String::new();
        print!(">>> ");

        let _ = std::io::stdout().flush();
        let _ = std::io::stdin().read_line(&mut line).unwrap();

        let input_chars = line.trim_end().chars().collect::<Vec<_>>();

        if input_chars.len() == 0 {
            continue 'loop_start;
        }

        let mut tokenizer = Tokenizer::new(&input_chars);
        let tokens = tokenizer.tokenize();

        match tokens {
            Ok(token_list) => {
                let mut parser = Parser::new(&token_list);
                let ast = parser.parse();

                match ast {
                    Ok(result) => {
                        println!("{:}", result.calculate());
                    }
                    Err(err) => {
                        println!("{:?}", err);
                        continue 'loop_start;
                    }
                }
            }
            Err(err) => {
                println!("{:?}", err);
                continue 'loop_start;
            }
        }
    }
}

#[derive(PartialEq, Debug, Clone, Copy)]
enum SymbolToken {
    SmallBracketOpen,
    SmallBracketClose,
}
impl SymbolToken {
    fn get_all() -> [SymbolToken; 2] {
        [Self::SmallBracketOpen, Self::SmallBracketClose]
    }
    fn value(&self) -> &str {
        match self {
            Self::SmallBracketOpen => "(",
            Self::SmallBracketClose => ")",
        }
    }
    fn is_equal<'a>(&self, check_with: &'a [char]) -> bool {
        check_with.iter().collect::<String>() == self.value()
    }
}
#[derive(PartialEq, Debug, Clone, Copy)]
enum OperatorToken {
    PLUS,
    MINUS,
    STAR,
    DIVIDE,
}
impl OperatorToken {
    fn get_all() -> [OperatorToken; 4] {
        [Self::PLUS, Self::MINUS, Self::STAR, Self::DIVIDE]
    }
    fn value(&self) -> &str {
        match self {
            Self::PLUS => "+",
            Self::MINUS => "-",
            Self::STAR => "*",
            Self::DIVIDE => "/",
        }
    }
    fn is_equal<'a>(&self, check_with: &'a [char]) -> bool {
        check_with.iter().collect::<String>() == self.value()
    }
}

#[derive(PartialEq, Debug, Clone, Copy)]
enum Token<'a> {
    Number(f64),
    Alphanumeric(&'a [char]),
    Operator(OperatorToken),
    Symbol(SymbolToken),
    WhiteSpace,
    EOF,
}

#[derive(PartialEq, Debug, Clone, Copy)]
struct TokenInfo {
    line_number: usize,
    column_number: usize,
}
struct Tokenizer<'a> {
    input_string: &'a [char],
    index: usize,
    len: usize,
    column_number: usize,
    line_number: usize,
}

impl<'a> Tokenizer<'a> {
    fn new(input_string: &'a [char]) -> Self {
        Self {
            input_string: input_string,
            index: 0,
            len: input_string.len(),
            column_number: 1,
            line_number: 1,
        }
    }
        }
    }

    fn advance(&mut self) {
        self.index += 1;
    }

    fn peek(&self) -> Option<&char> {
        if self.index < self.len {
            Some(&self.input_string[self.index])
        } else {
            None
        }
    }

    fn reset_index_at(&mut self, new_index: usize) {
        self.index = new_index;
    }

    fn get_alphanumeric_token(&self, start_index: usize, end_index: usize) -> Option<Token<'a>> {
        if start_index >= end_index {
            None
        } else {
            Some(Token::Alphanumeric(
                &self.input_string[start_index..end_index],
            ))
        }
    }
    fn match_any_in_blob(&mut self, blob: &str) -> Option<Token<'a>> {
        let initial = self.index;
        match self.peek() {
            Some(&chr) if blob.contains(chr) => {
                self.advance();
                return self.get_alphanumeric_token(initial, self.index);
            }
            _ => {
                return None;
            }
        }
    }
    fn match_many_in_blob(&mut self, blob: &str) -> Option<Token<'a>> {
        let initial_index = self.index;
        loop {
            match self.peek() {
                Some(&chr) if blob.contains(chr) => self.advance(),
                _ => break,
            }
        }
        self.get_alphanumeric_token(initial_index, self.index)
    }
    fn match_exact_in_blob(&mut self, blob: &str) -> Option<Token<'a>> {
        let initial_index = self.index;
        for c in blob.chars() {
            match self.peek() {
                Some(&chr) if chr == c => self.advance(),
                _ => {
                    self.reset_index_at(initial_index);
                    return None;
                }
            }
        }
        self.get_alphanumeric_token(initial_index, self.index)
    }
    fn match_number(&mut self) -> Option<Token<'a>> {
        let valid_numbers = "0123456789";
        match self.match_many_in_blob(valid_numbers) {
            Some(Token::Alphanumeric(a)) if a[0] != '0' => {
                Some(Token::Number(a.iter().collect::<String>().parse().unwrap()))
            }
            _ => None,
        }
    }
    fn match_operator(&mut self) -> Option<Token<'a>> {
        for operator_token in OperatorToken::get_all() {
            match self.match_exact_in_blob(operator_token.value()) {
                Some(Token::Alphanumeric(a)) if operator_token.is_equal(a) => {
                    return Some(Token::Operator(operator_token));
                }
                _ => {
                    continue;
                }
            }
        }
        None
    }
    fn match_symbol(&mut self) -> Option<Token<'a>> {
        for symbol_token in SymbolToken::get_all() {
            match self.match_exact_in_blob(symbol_token.value()) {
                Some(Token::Alphanumeric(a)) if symbol_token.is_equal(a) => {
                    return Some(Token::Symbol(symbol_token));
                }
                _ => {
                    continue;
                }
            }
        }
        None
    }
    fn match_whitespace(&mut self) -> Option<Token<'a>> {
        match self.match_many_in_blob(" ") {
            Some(_) => Some(Token::WhiteSpace),
            _ => None,
        }
    }

    fn tokenize(&mut self) -> Result<Vec<Token>, String> {
        let mut tokens = Vec::new();
        while self.peek() != None {
            if let Some(_) = self.match_whitespace() {
                continue;
            }
            if let Some(token) = self.match_operator() {
                tokens.push(token);
                continue;
            }
            if let Some(token) = self.match_number() {
                tokens.push(token);
                continue;
            }
            if let Some(token) = self.match_symbol() {
                tokens.push(token);
                continue;
            }
            return Result::Err(format!(
                "Unknown Token (Tokenizing Error) at index: {}",
                self.index
            ));
        }
        tokens.push(Token::EOF);
        return Result::Ok(tokens);
    }
}

struct Parser<'a> {
    tokens: &'a [Token<'a>],
    index: usize,
    len: usize,
}

impl<'a> Parser<'a> {
    fn new(tokens: &'a [Token<'a>]) -> Self {
        Parser {
            tokens: tokens,
            index: 0,
            len: tokens.len(),
        }
    }

    fn parse(&mut self) -> Result<Either, String> {
        let return_value = self.parse_term()?;
        match self.peek() {
            Some(Token::EOF) => {
                self.advance();
                Ok(return_value)
            }
            _ => Err(format!("Parsing Error at: {:} ", self.index)),
        }
    }

    fn advance(&mut self) -> Option<&Token> {
        if self.index < self.len {
            let return_value = &self.tokens[self.index];
            self.index += 1;
            Some(return_value)
        } else {
            None
        }
    }

    fn peek(&self) -> Option<&Token<'a>> {
        if self.index < self.len {
            Some(&self.tokens[self.index])
        } else {
            None
        }
    }

    fn parse_term(&mut self) -> Result<Either, String> {
        let mut first_expression = self.parse_prod()?;

        loop {
            match self.peek().cloned() {
                Some(Token::Operator(
                    operator_token @ (OperatorToken::PLUS | OperatorToken::MINUS),
                )) => {
                    self.advance();
                    let expr =
                        BinaryExpression::new(first_expression, self.parse_prod()?, operator_token);
                    first_expression = expr;
                    continue;
                }
                _ => {
                    return Ok(first_expression);
                }
            }
        }
    }

    fn parse_prod(&mut self) -> Result<Either, String> {
        let mut first_expression = self.parse_unary()?;

        loop {
            match self.peek().cloned() {
                Some(Token::Operator(
                    operator_token @ (OperatorToken::STAR | OperatorToken::DIVIDE),
                )) => {
                    self.advance();
                    let expr =
                        BinaryExpression::new(first_expression, self.parse_prod()?, operator_token);
                    first_expression = expr;
                    continue;
                }
                _ => {
                    return Ok(first_expression);
                }
            }
        }
    }

    fn parse_unary(&mut self) -> Result<Either, String> {
        match self.peek().cloned() {
            Some(Token::Operator(operator @ (OperatorToken::PLUS | OperatorToken::MINUS))) => {
                self.advance();
                let expr = UnaryExpression::new(self.parse_unary()?, operator);
                Ok(expr)
            }
            _ => self.parse_bracket(),
        }
    }

    fn parse_bracket(&mut self) -> Result<Either, String> {
        match self.peek().cloned() {
            Some(Token::Symbol(SymbolToken::SmallBracketOpen)) => {
                self.advance();
                let val = self.parse_term()?;
                match self.peek().cloned() {
                    Some(Token::Symbol(SymbolToken::SmallBracketClose)) => {
                        self.advance();
                        Ok(val)
                    }
                    _ => Err(format!("Error: No closing bracket found: {}", self.index)),
                }
            }
            _ => self.parse_number(),
        }
    }

    fn parse_number(&mut self) -> Result<Either, String> {
        match self.advance() {
            Some(&Token::Number(val)) => Ok(Either::Number(val)),
            _ => Err(format!(
                "Error parsing Number at token index {}",
                self.index
            )),
        }
    }
}

#[derive(Debug)]
enum Either {
    Number(f64),
    BinaryExpression(Box<BinaryExpression>),
    UnaryExpression(Box<UnaryExpression>),
}

impl Either {
    fn calculate(&self) -> f64 {
        match self {
            Self::Number(val) => *val,

            Self::BinaryExpression(boxed_expression) => match &**boxed_expression {
                BinaryExpression {
                    left,
                    right,
                    operator,
                } => match operator {
                    OperatorToken::PLUS => left.calculate() + right.calculate(),
                    OperatorToken::MINUS => left.calculate() - right.calculate(),
                    OperatorToken::STAR => left.calculate() * right.calculate(),
                    OperatorToken::DIVIDE => left.calculate() / right.calculate(),
                },
            },
            Self::UnaryExpression(boxed_expression) => match &**boxed_expression {
                UnaryExpression { val, operator } => match operator {
                    OperatorToken::PLUS => 1.0 * val.calculate(),
                    OperatorToken::MINUS | _ => -1.0 * val.calculate(),
                },
            },
        }
    }
}

#[derive(Debug)]
struct BinaryExpression {
    left: Either,
    right: Either,
    operator: OperatorToken,
}
impl BinaryExpression {
    fn new(left: Either, right: Either, operator: OperatorToken) -> Either {
        Either::BinaryExpression(Box::new(BinaryExpression {
            left: left,
            right: right,
            operator: operator,
        }))
    }
}
#[derive(Debug)]
struct UnaryExpression {
    val: Either,
    operator: OperatorToken,
}
impl UnaryExpression {
    fn new(val: Either, operator: OperatorToken) -> Either {
        Either::UnaryExpression(Box::new(UnaryExpression {
            val: val,
            operator: operator,
        }))
    }
}

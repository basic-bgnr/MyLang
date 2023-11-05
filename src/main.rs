use std::io::Write;

fn main() {
    test();
}

fn test() {
    // let input = "3*3-6";
    // let (tokens, _) = lexer_generate_token(input);
    // println!("{:?}", tokens);
    // let mut parser = Parser::new(&tokens);
    // let expr = parser.parse_term();
    // println!("{:} = {:?}", input, expr.calculate());
    loop {
        let mut line = String::new();
        print!(">>> ");
        let _ = std::io::stdout().flush();
        let _ = std::io::stdin().read_line(&mut line).unwrap();
        let (tokens, _) = lexer_generate_token(&line);
        let mut parser = Parser::new(&tokens);
        let expr = parser.parse_term();
        println!("{:}", expr.calculate());
    }
}

#[derive(PartialEq, Debug, Clone, Copy)]
enum Operator {
    PLUS,
    MINUS,
    STAR,
    DIVIDE,
}

#[derive(PartialEq, Debug)]
enum Token<'a> {
    Number(f64),
    String(&'a str),
    Operator(Operator),
}

type ParsedToken<'a> = (Option<Token<'a>>, &'a str);

fn match_integer<'a>(input: &'a str) -> ParsedToken<'a> {
    let mut index = 0;
    for chr in input.chars() {
        if !('0'..='9').contains(&chr) {
            break;
        }
        index += 1;
    }
    match (&input[0..index]).parse() {
        Ok(num) => (Some(Token::Number(num)), &input[index..input.len()]),
        Err(_) => (None, &input),
    }
}

fn match_alphabet<'a>(input: &'a str) -> ParsedToken<'a> {
    let mut index = 0;
    for chr in input.chars() {
        if !('A'..='Z').contains(&chr) && !('a'..='z').contains(&chr) {
            break;
        }
        index += 1;
    }
    match index {
        0 => (None, &input[index..input.len()]),
        _ => (
            Some(Token::String(&input[0..index])),
            &input[index..input.len()],
        ),
    }
}

fn match_operator<'a>(input: &'a str) -> ParsedToken<'a> {
    match input.chars().next() {
        Some('+') => (
            Some(Token::Operator(Operator::PLUS)),
            &input[1..input.len()],
        ),
        Some('-') => (
            Some(Token::Operator(Operator::MINUS)),
            &input[1..input.len()],
        ),
        Some('/') => (
            Some(Token::Operator(Operator::DIVIDE)),
            &input[1..input.len()],
        ),
        Some('*') => (
            Some(Token::Operator(Operator::STAR)),
            &input[1..input.len()],
        ),
        _ => (None, input),
    }
}

fn match_optional<'a>(input: &'a str) -> ParsedToken {
    let (parsed, remaining) = match_integer(input);
    if parsed != None {
        return (parsed, remaining);
    }

    let (parsed, remaining) = match_operator(input);
    if parsed != None {
        return (parsed, remaining);
    }

    match_alphabet(input)
}

fn lexer_generate_token<'a>(mut input: &'a str) -> (Vec<Token>, &'a str) {
    let mut output = Vec::new();
    loop {
        let (parsed, remaining) = match_optional(input);

        if parsed != None {
            output.push(parsed.unwrap());
        } else {
            return (output, remaining);
        }
        if remaining == "" {
            return (output, remaining);
        }
        input = remaining;
    }
}

#[derive(Debug)]
enum Either {
    Number(f64),
    Expression(Box<BinaryExpression>),
}
impl Either {
    fn calculate(&self) -> f64 {
        match self {
            Self::Number(val) => *val,

            Self::Expression(boxed_expression) => match &**boxed_expression {
                BinaryExpression {
                    left,
                    right,
                    operator,
                } => match operator {
                    Operator::PLUS => left.calculate() + right.calculate(),
                    Operator::MINUS => left.calculate() - right.calculate(),
                    Operator::STAR => left.calculate() * right.calculate(),
                    Operator::DIVIDE => left.calculate() / right.calculate(),
                },
            },
        }
    }
}

#[derive(Debug)]
struct BinaryExpression {
    left: Either,
    right: Either,
    operator: Operator,
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

    fn advance(&mut self) -> Option<&Token> {
        if self.index < self.len {
            let return_value = &self.tokens[self.index];
            self.index += 1;
            Some(return_value)
        } else {
            None
        }
    }

    fn peek_and_match(&self, token: &Token) -> bool {
        if self.index < self.len {
            &self.tokens[self.index] == token
        } else {
            false
        }
    }

    fn parse_term(&mut self) -> Either {
        let mut first_expression = self.parse_prod();

        loop {
            if self.peek_and_match(&Token::Operator(Operator::PLUS)) {
                self.advance();
                let expr = Either::Expression(Box::new(BinaryExpression {
                    left: first_expression,
                    right: self.parse_prod(),
                    operator: Operator::PLUS,
                }));
                first_expression = expr;
            } else if self.peek_and_match(&Token::Operator(Operator::MINUS)) {
                self.advance();
                let expr = Either::Expression(Box::new(BinaryExpression {
                    left: first_expression,
                    right: self.parse_prod(),
                    operator: Operator::MINUS,
                }));
                first_expression = expr;
            } else {
                return first_expression;
            }
        }
    }

    fn parse_prod(&mut self) -> Either {
        let mut first_expression = self.parse_number();

        loop {
            if self.peek_and_match(&Token::Operator(Operator::STAR)) {
                self.advance();
                let expr = Either::Expression(Box::new(BinaryExpression {
                    left: first_expression,
                    right: self.parse_number(),
                    operator: Operator::STAR,
                }));
                first_expression = expr;
            } else if self.peek_and_match(&Token::Operator(Operator::DIVIDE)) {
                self.advance();
                let expr = Either::Expression(Box::new(BinaryExpression {
                    left: first_expression,
                    right: self.parse_number(),
                    operator: Operator::DIVIDE,
                }));
                first_expression = expr;
            } else {
                return first_expression;
            }
        }
    }

    fn parse_number(&mut self) -> Either {
        match self.advance() {
            Some(&Token::Number(val)) => Either::Number(val),
            _ => panic!("parsing error"),
        }
    }
}

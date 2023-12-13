use std::{collections::HashMap, format, io::Write, println, unimplemented, unreachable, vec};

fn main() {
    let prompt = "
    let a = 1
    let b = 1
    let counter = 1
    while {counter < 10} {
                            let c = a + b
                            a = b
                            b = c
                            counter = counter + 1
                         }
    counter a b
    ";
    println!("Fibonacci Calculator:\n{}", prompt);

    let mut interpreter = Interpreter::new();

    interpreter.interpret(prompt);
    interpreter.interactive_prompt();
}

#[derive(PartialEq, Debug, Clone, Copy)]
enum KeywordToken {
    Let,
    True,
    False,
    While,
    If,
    Else,
}
impl KeywordToken {
    fn get_all() -> [KeywordToken; 6] {
        [
            Self::Let,
            Self::True,
            Self::False,
            Self::While,
            Self::If,
            Self::Else,
        ]
    }
    fn value(&self) -> &str {
        match self {
            Self::Let => "let",
            Self::True => "true",
            Self::False => "false",
            Self::While => "while",
            Self::If => "if",
            Self::Else => "else",
        }
    }
    fn is_equal<'a>(&self, check_with: &'a [char]) -> bool {
        check_with.iter().collect::<String>() == self.value()
    }
}

#[derive(PartialEq, Debug, Clone, Copy)]
enum SymbolToken {
    SmallBracketOpen,
    SmallBracketClose,
    Dot,
    Equal,
    // UnderScore,
    SemiColon,
    CurlyBracketOpen,
    CurlyBracketClose,
}
impl SymbolToken {
    fn get_all() -> [SymbolToken; 7] {
        [
            Self::SmallBracketOpen,
            Self::SmallBracketClose,
            Self::Dot,
            Self::Equal,
            // Self::UnderScore,
            Self::SemiColon,
            Self::CurlyBracketOpen,
            Self::CurlyBracketClose,
        ]
    }
    fn value(&self) -> &str {
        match self {
            Self::SmallBracketOpen => "(",
            Self::SmallBracketClose => ")",
            Self::Dot => ".",
            Self::Equal => "=",
            // Self::UnderScore => "_",
            Self::SemiColon => ";",
            Self::CurlyBracketOpen => "{",
            Self::CurlyBracketClose => "}",
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

    LOGICAL_AND,
    LOGICAL_OR,

    LOGICAL_NOT,

    LESS_THAN,
    GREATER_THAN,
    EQUAL_TO,
}
impl OperatorToken {
    fn get_all() -> [OperatorToken; 10] {
        [
            Self::PLUS,
            Self::MINUS,
            Self::STAR,
            Self::DIVIDE,
            Self::LOGICAL_AND,
            Self::LOGICAL_OR,
            Self::LOGICAL_NOT,
            Self::LESS_THAN,
            Self::GREATER_THAN,
            Self::EQUAL_TO,
        ]
    }
    fn value(&self) -> &str {
        match self {
            Self::PLUS => "+",
            Self::MINUS => "-",
            Self::STAR => "*",
            Self::DIVIDE => "/",
            Self::LOGICAL_AND => "&&",
            Self::LOGICAL_OR => "||",
            Self::LOGICAL_NOT => "!",
            Self::LESS_THAN => "<",
            Self::GREATER_THAN => ">",
            Self::EQUAL_TO => "==",
        }
    }
    //output type after application of operator
    fn tipe(&self) -> &LanguageType {
        match self {
            Self::PLUS | Self::MINUS | Self::STAR | Self::DIVIDE => &LanguageType::Number,
            Self::LOGICAL_AND
            | Self::LOGICAL_OR
            | Self::LOGICAL_NOT
            | Self::LESS_THAN
            | Self::GREATER_THAN
            | Self::EQUAL_TO => &LanguageType::Boolean,
        }
    }
    fn is_equal<'a>(&self, check_with: &'a [char]) -> bool {
        check_with.iter().collect::<String>() == self.value()
    }
}

#[derive(PartialEq, Debug, Clone, Copy)]
enum Token<'a> {
    Digit(&'a char, TokenInfo),
    Alphabet(&'a char, TokenInfo),
    Identifier(&'a [char], TokenInfo),

    Operator(OperatorToken, TokenInfo),
    Symbol(SymbolToken, TokenInfo),

    Keyword(KeywordToken, TokenInfo),

    WhiteSpace(TokenInfo),
    NewLine(TokenInfo),

    EOF(TokenInfo),

    Alphanumeric(&'a [char]),
}
impl<'a> Token<'a> {
    fn get_token_info(&self) -> &TokenInfo {
        match self {
            Self::Digit(_, token_info) => token_info,
            Self::Alphabet(_, token_info) => token_info,
            Self::Identifier(_, token_info) => token_info,
            Self::Operator(_, token_info) => token_info,
            Self::Symbol(_, token_info) => token_info,
            Self::Keyword(_, token_info) => token_info,
            Self::EOF(token_info) => token_info,

            Self::WhiteSpace(token_info) => token_info,
            Self::NewLine(token_info) => token_info,
            Self::Alphanumeric(_) => unreachable!(),
        }
    }
    fn value(&self) -> String {
        match self {
            Self::Digit(d, _) => d.to_string(),
            Self::Alphabet(a, _) => a.to_string(),
            Self::Identifier(i, _) => i.iter().collect::<String>(),
            Self::Operator(o, _) => o.value().to_string(),
            Self::Symbol(s, _) => s.value().to_string(),
            Self::Keyword(k, _) => k.value().to_string(),

            Self::WhiteSpace(_) => Token::value_of_white_space().to_string(),
            Self::NewLine(_) => Token::value_of_new_line().to_string(),
            Self::Alphanumeric(lst_chr) => lst_chr.iter().collect(),
            Self::EOF(_) => Token::value_of_eof().to_string(),
        }
    }
    fn value_of_white_space() -> &'static str {
        " "
    }
    fn value_of_new_line() -> &'static str {
        "\n"
    }
    fn value_of_eof() -> &'static str {
        "<EOF>"
    }
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
    fn get_token_info(&self) -> TokenInfo {
        TokenInfo {
            line_number: self.line_number,
            column_number: self.column_number,
        }
    }

    fn advance(&mut self) {
        self.index += 1;
        self.column_number += 1;
    }
    fn break_lines(&mut self, lines: usize) {
        self.line_number += lines;
        self.reset_column();
    }

    fn peek(&self) -> Option<&char> {
        if self.index < self.len {
            Some(&self.input_string[self.index])
        } else {
            None
        }
    }

    fn reset_index_at(&mut self, new_index: usize) {
        let index_diff = self.index - new_index;
        self.index = new_index;
        self.column_number -= index_diff;
    }
    fn reset_column(&mut self) {
        self.column_number = 1;
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

    fn match_operator(&mut self) -> Option<Token<'a>> {
        let token_info = self.get_token_info();
        for operator_token in OperatorToken::get_all() {
            match self.match_exact_in_blob(operator_token.value()) {
                Some(Token::Alphanumeric(a)) if operator_token.is_equal(a) => {
                    return Some(Token::Operator(operator_token, token_info));
                }
                _ => {
                    continue;
                }
            }
        }
        None
    }
    fn match_symbol(&mut self) -> Option<Token<'a>> {
        let token_info = self.get_token_info();
        for symbol_token in SymbolToken::get_all() {
            match self.match_exact_in_blob(symbol_token.value()) {
                Some(Token::Alphanumeric(a)) if symbol_token.is_equal(a) => {
                    return Some(Token::Symbol(symbol_token, token_info));
                }
                _ => {
                    continue;
                }
            }
        }
        None
    }
    fn match_whitespace(&mut self) -> Option<Token<'a>> {
        let token_info = self.get_token_info();
        match self.match_many_in_blob(&Token::value_of_white_space()) {
            Some(_) => Some(Token::WhiteSpace(token_info)),
            _ => None,
        }
    }
    fn match_new_lines(&mut self) -> Option<Token<'a>> {
        let token_info = self.get_token_info();
        match self.match_many_in_blob(&Token::value_of_new_line()) {
            Some(new_lines) => {
                let no_of_new_lines = new_lines.value().len();
                self.break_lines(no_of_new_lines);

                Some(Token::NewLine(token_info))
            }
            _ => None,
        }
    }

    fn match_digit(&mut self) -> Option<Token<'a>> {
        let token_info = self.get_token_info();
        match self.match_any_in_blob("0123456789") {
            Some(Token::Alphanumeric(chr)) => Some(Token::Digit(chr.first().unwrap(), token_info)),
            _ => None,
        }
    }
    fn match_alphabet(&mut self) -> Option<Token<'a>> {
        let blob = ('a'..='z').chain('A'..='Z').collect::<String>();
        let token_info = self.get_token_info();
        match self.match_any_in_blob(&blob) {
            Some(Token::Alphanumeric(chr)) => {
                Some(Token::Alphabet(chr.first().unwrap(), token_info))
            }
            _ => None,
        }
    }
    fn match_identifier(&mut self) -> Option<Token<'a>> {
        let blob = ('a'..='z').chain('A'..='Z').collect::<String>();
        let token_info = self.get_token_info();
        match self.match_many_in_blob(&blob) {
            Some(Token::Alphanumeric(chr)) => Some(Token::Identifier(chr, token_info)),
            _ => None,
        }
    }
    fn match_keyword(&mut self) -> Option<Token<'a>> {
        let token_info = self.get_token_info();
        let initial_index = self.index;
        for keyword_token in KeywordToken::get_all() {
            match self.match_exact_in_blob(keyword_token.value()) {
                Some(Token::Alphanumeric(_)) => match self.match_alphabet() {
                    Some(_) => {
                        self.reset_index_at(initial_index);
                        return None;
                    }
                    None => {
                        return Some(Token::Keyword(keyword_token, token_info));
                    }
                },
                _ => {
                    continue;
                }
            }
        }
        None
    }

    fn tokenize(&mut self) -> Result<Vec<Token>, Vec<String>> {
        let mut tokens = Vec::new();
        let mut tokenization_errors = vec![];
        while self.peek().is_some() {
            if let Some(token) = self.match_keyword() {
                tokens.push(token);
                continue;
            }
            if let Some(token) = self.match_digit() {
                tokens.push(token);
                continue;
            }
            if let Some(token) = self.match_identifier() {
                tokens.push(token);
                continue;
            }
            if let Some(token) = self.match_operator() {
                tokens.push(token);
                continue;
            }
            if let Some(token) = self.match_symbol() {
                tokens.push(token);
                continue;
            }
            if let Some(token) = self.match_whitespace() {
                tokens.push(token);
                continue;
            }
            if let Some(token) = self.match_new_lines() {
                // tokens.push(token);
                continue;
            }
            let token_info = self.get_token_info();
            tokenization_errors.push(format!(
                "Tokenizing Error: Unknown token at line {}, column {}",
                token_info.line_number, token_info.column_number
            ));
            self.advance();
        }
        if tokenization_errors.len() > 0 {
            Err(tokenization_errors)
        } else {
            let token_info = self.get_token_info();
            tokens.push(Token::EOF(token_info));
            // println!("{:?}", tokens);
            Result::Ok(tokens)
        }
    }
}

struct Parser<'a> {
    tokens: &'a [Token<'a>],
    index: usize,
    back_track_index: usize,
    len: usize,
    block_position: usize,
    num_identifiers_in_block: Vec<usize>,
    identifiers_list: Option<Vec<Identifier>>,
}

impl<'a> Parser<'a> {
    fn new(tokens: &'a [Token<'a>]) -> Self {
        Parser {
            tokens: tokens,
            index: 0,
            back_track_index: 0,
            len: tokens.len(),
            block_position: 0,
            num_identifiers_in_block: vec![0],
            identifiers_list: None,
        }
    }
    fn increment_num_identifier_in_current_block(&mut self) {
        let block_postion = self.get_block_position();
        self.num_identifiers_in_block[block_postion] += 1;
    }

    fn get_block_position(&self) -> usize {
        self.block_position
    }
    fn increment_block_position(&mut self) {
        self.block_position += 1;
        self.num_identifiers_in_block.push(0)
    }
    fn decrement_block_position(&mut self) {
        let current_block_position = self.block_position;
        let num_identifiers_to_pop = self.num_identifiers_in_block[current_block_position];
        for _ in 0..num_identifiers_to_pop {
            self.identifiers_list.as_mut().unwrap().pop();
        }
        self.block_position -= 1;
        self.num_identifiers_in_block.pop();
    }

    fn remember_index(&mut self) {
        self.back_track_index = self.index;
    }

    fn back_track(&mut self) {
        self.index = self.back_track_index;
    }

    fn reset_index_at(&mut self, index: usize) {
        self.index = index;
    }

    fn get_identifiers_list(self) -> Option<Vec<Identifier>> {
        self.identifiers_list
    }

    fn push_identifier(&mut self, identifier: Identifier) {
        if self.identifiers_list.is_none() {
            self.identifiers_list = Some(vec![identifier]);
        } else {
            self.identifiers_list.as_mut().unwrap().push(identifier);
        }

        self.increment_num_identifier_in_current_block();
    }

    fn set_previous_identifiers_list(&mut self, previous_identifier_list: Option<Vec<Identifier>>) {
        self.identifiers_list = previous_identifier_list
    }

    fn get_reference_to_identifier(&self, name: &str) -> Option<Identifier> {
        match &self.identifiers_list {
            Some(identifiers) => {
                for identifier in identifiers.iter().rev() {
                    if name == identifier.name {
                        return Some(identifier.clone());
                    }
                }
            }
            None => return None,
        }
        None
    }

    fn modify_identifier_type(
        &mut self,
        identifier_to_modify: &Identifier,
        tipe_value: LanguageType,
    ) {
        match self.identifiers_list.as_deref_mut() {
            Some(identifiers) => {
                for mut identifier in identifiers.iter_mut().rev() {
                    if identifier.name == identifier_to_modify.name {
                        *identifier = Identifier::new(
                            &identifier.name,
                            identifier.block_position,
                            tipe_value,
                        );
                    }
                }
            }
            _ => {}
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

    fn consume_optional_semicolon(&mut self) -> bool {
        if let Some(Token::Symbol(SymbolToken::SemiColon, _)) = self.peek() {
            self.advance();
            return true;
        }
        false
    }

    fn consume_optional_whitespace(&mut self) -> bool {
        if let Some(Token::WhiteSpace(_)) = self.peek() {
            self.advance();
            return true;
        }
        false
    }

    fn parse(&mut self) -> Result<Vec<Either>, Error> {
        let mut statements = Vec::new();
        loop {
            let parsed_statement = self.parse_next_statement()?;
            self.consume_optional_semicolon();
            self.consume_optional_whitespace();
            statements.push(parsed_statement);

            if self.peek().is_none() {
                break;
            }
        }
        // println!("{:?}", statements);

        Ok(statements)
    }

    fn parse_next_statement(&mut self) -> Result<Either, Error> {
        self.parse_halt_statement()
            .or_else(|e| match e {
                Error::ParseError(_) => self.parse_while_statement(),
                Error::TypeError(_) | Error::IdentifierError(_) | Error::SyntaxError(_) => Err(e),
            })
            .or_else(|e| match e {
                Error::ParseError(_) => self.parse_let_statement(),
                Error::TypeError(_) | Error::IdentifierError(_) | Error::SyntaxError(_) => Err(e),
            })
            .or_else(|e| match e {
                Error::ParseError(_) => self.parse_assignment_statement(),
                Error::TypeError(_) | Error::IdentifierError(_) | Error::SyntaxError(_) => Err(e),
            })
            .or_else(|e| match e {
                Error::ParseError(_) => self.parse_expression(),
                Error::TypeError(_) | Error::IdentifierError(_) | Error::SyntaxError(_) => Err(e),
            })
            .or_else(|e| match e {
                e @ (Error::ParseError(_)
                | Error::TypeError(_)
                | Error::IdentifierError(_)
                | Error::SyntaxError(_)) => Err(e),
            })
            .map(|(statement, _)| statement)
    }
    fn parse_halt_statement(&mut self) -> Result<(Either, TokenInfo), Error> {
        match self.peek().cloned() {
            Some(Token::EOF(token_info)) => {
                self.advance();
                Ok((Either::HaltStatement, token_info))
            }
            Some(token) => Err(Error::ParseError(format!(
                "Parsing Error: Cannot parse while statement found at line {}, column {}",
                token.get_token_info().line_number,
                token.get_token_info().column_number,
            ))),
            None => unreachable!(),
        }
    }
    fn parse_while_statement(&mut self) -> Result<(Either, TokenInfo), Error> {
        self.consume_optional_whitespace();
        match self.peek().cloned() {
            Some(Token::Keyword(keyword, token_info_while)) if keyword == KeywordToken::While => {
                self.advance();
                self.consume_optional_whitespace();
                let (condition_expression, token_info) = self.parse_logical_term()?;
                if condition_expression.tipe() == &LanguageType::Boolean {
                    self.consume_optional_whitespace();
                    let (block_expression, _) = self.parse_logical_term()?;
                    let tipe = block_expression.tipe().clone();
                    Ok((
                        Either::WhileStatement(
                            Box::new(condition_expression),
                            Box::new(block_expression),
                        ),
                        token_info_while,
                    ))
                } else {
                    Err(Error::ParseError(format!(
                        "Type Error: Cannot parse while statement found at line {}, column {}",
                        token_info.line_number, token_info.column_number
                    )))
                }
            }
            Some(token) => Err(Error::ParseError(format!(
                "Parsing Error: Cannot parse while statement found at line {}, column {}",
                token.get_token_info().line_number,
                token.get_token_info().column_number,
            ))),
            _ => unreachable!(),
        }
    }

    fn parse_let_statement(&mut self) -> Result<(Either, TokenInfo), Error> {
        self.consume_optional_whitespace();
        match self.peek().cloned() {
            Some(Token::Keyword(keyword, token_info))
                if keyword.value() == KeywordToken::Let.value() =>
            {
                self.advance();
                self.advance(); //consume whitespace
                match self.peek().cloned() {
                    Some(identifier @ Token::Identifier(_, token_info)) => {
                        self.advance();
                        self.consume_optional_whitespace();
                        match self.peek().cloned() {
                            Some(Token::Symbol(symbol, _))
                                if symbol.value() == SymbolToken::Equal.value() =>
                            {
                                self.advance();
                                self.consume_optional_whitespace();
                                let (term, token_info) = self.parse_expression()?;

                                let tipe = term.tipe();

                                match tipe {
                                    LanguageType::Void => Err(Error::TypeError(format!(
                                        "Type Error: Void type found at line {}, column {}",
                                        token_info.line_number, token_info.column_number
                                    ))),
                                    _ => {
                                        self.push_identifier(Identifier::new(
                                            &identifier.value(),
                                            self.get_block_position(),
                                            *tipe,
                                        ));

                                        Ok((
                                            LetStatement::new(identifier.value(), term),
                                            token_info,
                                        ))
                                    }
                                }
                            }
                            Some(token) => {
                                self.push_identifier(Identifier::new(
                                    &identifier.value(),
                                    self.get_block_position(),
                                    LanguageType::Void,
                                ));

                                Ok((
                                    LetStatement::new(
                                        identifier.value(),
                                        Either::Placeholder(LanguageType::Untyped),
                                    ),
                                    token_info,
                                ))
                            }
                            Some(token) => Err(Error::SyntaxError(format!(
                                "Syntax Error: No equal found at line {}, column {}",
                                token.get_token_info().line_number,
                                token.get_token_info().column_number
                            ))),
                            None => Err(Error::SyntaxError(format!(
                                "Parsing Error: None value encountered at line {}, {}",
                                token_info.line_number, token_info.column_number
                            ))),
                        }
                    }
                    Some(token) => Err(Error::SyntaxError(format!(
                        "Syntax Error: No identifier found at line {}, column {}",
                        token.get_token_info().line_number,
                        token.get_token_info().column_number
                    ))),
                    None => Err(Error::SyntaxError(format!(
                        "Syntax Error: None value found at line {}, column {}",
                        token_info.line_number, token_info.column_number
                    ))),
                }
            }
            Some(token) => Err(Error::ParseError(format!(
                "Parsing Error: Cannot parse let statement at line {}, column {}",
                token.get_token_info().line_number,
                token.get_token_info().column_number
            ))),
            _ => unreachable!(),
        }
    }
    fn parse_assignment_statement(&mut self) -> Result<(Either, TokenInfo), Error> {
        self.consume_optional_whitespace();
        self.remember_index();
        match self.peek().cloned() {
            Some(identifier_token @ Token::Identifier(_, token_info)) => {
                match self.get_reference_to_identifier(&identifier_token.value()) {
                    Some(identifier) => {
                        self.advance();
                        self.consume_optional_whitespace();
                        match self.peek().cloned() {
                            Some(Token::Symbol(symbol, _))
                                if symbol.value() == SymbolToken::Equal.value() =>
                            {
                                self.advance();
                                self.consume_optional_whitespace();
                                let (term, token_info) = self.parse_expression()?;
                                if term.tipe() == identifier.tipe() || identifier.tipe() == &LanguageType::Void{
                                    self.modify_identifier_type(&identifier, *term.tipe());
                                    Ok((
                                        AssignmentStatement::new(identifier_token.value(), term),
                                        token_info,
                                    ))
                                } else {
                                    Err(Error::TypeError(format!(
                                        "Type Error: Type mismatched, found at line {}, column {}",
                                        token_info.line_number, token_info.column_number
                                    )))
                                }
                            }
                            Some(other_token) => {
                                self.back_track();
                                Err(Error::ParseError(format!(
                                "Parsing Error: Cannot parse assignment statement found at line {}, column {}",
                                other_token.get_token_info().line_number,
                                other_token.get_token_info().column_number
                            )))
                            }
                            None => Err(Error::ParseError(format!(
                                "Parsing Error: Cannot parse assignment statement found at line {}, column {}",
                                token_info.line_number,
                                token_info.column_number
                            ))),
                        }
                    }
                    None => Err(Error::IdentifierError(format!(
                        "Identifier Error: use of undeclared variable '{}' encountered at line {}, column {}",
                        identifier_token.value(),
                        token_info.line_number,
                        token_info.column_number
                    ))),
                }
            }
            Some(other_token) => Err(Error::ParseError(format!(
                "Parsing Error: Cannot parse assignment statement at line {}, column {}",
                other_token.get_token_info().line_number,
                other_token.get_token_info().column_number
            ))),
            _ => unreachable!(),
        }
    }

    fn parse_expression(&mut self) -> Result<(Either, TokenInfo), Error> {
        self.parse_if_else_expression().or_else(|e| match e {
            Error::ParseError(_) => self.parse_logical_term(),
            Error::TypeError(_) | Error::IdentifierError(_) | Error::SyntaxError(_) => Err(e),
        })
        // .or_else(|e| match e {
        //     e @ (Error::ParseError(_)
        //     | Error::TypeError(_)
        //     | Error::IdentifierError(_)
        //     | Error::SyntaxError(_)) => Err(e),
        // })
    }

    fn parse_if_else_expression(&mut self) -> Result<(Either, TokenInfo), Error> {
        self.consume_optional_whitespace();
        match self.peek().cloned() {
            Some(Token::Keyword(keyword, token_info_if)) if keyword == KeywordToken::If => {
                self.advance();
                self.consume_optional_whitespace();
                let (condition_expression, token_info) = self.parse_logical_term()?;
                if condition_expression.tipe() == &LanguageType::Boolean {
                    self.consume_optional_whitespace();
                    let (if_block_expression, _) = self.parse_logical_term()?;
                    let if_block_tipe = if_block_expression.tipe().clone();

                    self.consume_optional_whitespace();

                    match self.peek().cloned(){
                        Some(Token::Keyword(keyword, token_info_while)) if keyword == KeywordToken::Else => {
                            self.advance();
                            self.consume_optional_whitespace();

                            let (else_block_expression, token_info) = self.parse_expression()?;
                            let else_block_tipe = else_block_expression.tipe();

                            if if_block_tipe == *else_block_tipe {
                                Ok((Either::IfElseStatement(Box::new(condition_expression), Box::new(if_block_expression), Box::new(else_block_expression), if_block_tipe), token_info_if))
                            }else{

                                    Err(Error::TypeError(format!( "Type Error: type mismatched of if and else block found at line {}, column {}",
                                token_info.line_number,
                                token_info.column_number
                            )))
                            }
                    }
                    Some(other_token) =>
                                    Err(Error::SyntaxError(format!( "Syntax Error: Cannot parse matching else block at line {}, column {}, found {} instead",
                                other_token.get_token_info().line_number,
                                other_token.get_token_info().column_number,
                                other_token.value(),
                            ))),
                                    _ => unreachable!(),
                }
                } else {
                    Err(Error::TypeError(format!(
                        "Type Error: Type mismatched found at line {}, column {}",
                        token_info.line_number, token_info.column_number
                    )))
                }
            }
            Some(token) => Err(Error::ParseError(format!(
                "Parsing Error: Cannot parse ifelse statement found at line {}, column {}",
                token.get_token_info().line_number,
                token.get_token_info().column_number,
            ))),
            _ => unreachable!(),
        }
    }

    fn parse_logical_term(&mut self) -> Result<(Either, TokenInfo), Error> {
        self.consume_optional_whitespace();
        let (mut first_expression, token_info) = self.parse_comparison_term()?;
        loop {
            match self.peek().cloned() {
                Some(Token::Operator(
                    operator_token @ (OperatorToken::LOGICAL_AND | OperatorToken::LOGICAL_OR),
                    token_info,
                )) if first_expression.tipe() == &LanguageType::Boolean => {
                    self.advance();
                    let (second_expression, token_info) = self.parse_comparison_term()?;
                    if first_expression.tipe() == second_expression.tipe() {
                        let expr = BinaryExpression::new(
                            first_expression,
                            second_expression,
                            operator_token,
                        );
                        first_expression = expr;
                        continue;
                    } else {
                        return Err(Error::TypeError(format!(
                            "Type Error: Type mismatched at line {}, column {}",
                            token_info.line_number, token_info.column_number
                        )));
                    }
                }
                Some(Token::Operator(
                    OperatorToken::LOGICAL_AND | OperatorToken::LOGICAL_OR,
                    token_info,
                )) if first_expression.tipe() == &LanguageType::Number => {
                    return Err(Error::TypeError(format!(
                        "Type Error: Operator mismatched at line {}, column {}",
                        token_info.line_number, token_info.column_number
                    )));
                }
                Some(Token::WhiteSpace(_)) => {
                    self.advance();
                    continue;
                }
                _ => {
                    return Ok((first_expression, token_info));
                }
            }
        }
    }
    fn parse_comparison_term(&mut self) -> Result<(Either, TokenInfo), Error> {
        let (mut first_expression, token_info) = self.parse_term()?;

        loop {
            match self.peek().cloned() {
                Some(Token::Operator(
                    operator_token @ (OperatorToken::LESS_THAN
                    | OperatorToken::GREATER_THAN
                    | OperatorToken::EQUAL_TO),
                    token_info,
                )) if first_expression.tipe() == &LanguageType::Number => {
                    self.advance();
                    let (second_expression, token_info) = self.parse_term()?;
                    if first_expression.tipe() == second_expression.tipe() {
                        let expr = BinaryExpression::new(
                            first_expression,
                            second_expression,
                            operator_token,
                        );
                        first_expression = expr;
                        continue;
                    } else {
                        return Err(Error::TypeError(format!(
                            "Type Error: Type mismatched at line {}, column {}",
                            token_info.line_number, token_info.column_number
                        )));
                    }
                }
                Some(Token::Operator(operator_token @ OperatorToken::EQUAL_TO, token_info))
                    if first_expression.tipe() == &LanguageType::Boolean =>
                {
                    self.advance();
                    let (second_expression, token_info) = self.parse_term()?;
                    if first_expression.tipe() == second_expression.tipe() {
                        let expr = BinaryExpression::new(
                            first_expression,
                            second_expression,
                            operator_token,
                        );
                        first_expression = expr;
                        continue;
                    } else {
                        return Err(Error::TypeError(format!(
                            "Type Error: Type mismatched at line {}, column {}",
                            token_info.line_number, token_info.column_number
                        )));
                    }
                }
                Some(Token::Operator(
                    operator_token @ (OperatorToken::LESS_THAN | OperatorToken::GREATER_THAN),
                    token_info,
                )) if first_expression.tipe() == &LanguageType::Boolean => {
                    return Err(Error::TypeError(format!(
                        "Type Error: Operator mismatched at line {}, column {}",
                        token_info.line_number, token_info.column_number
                    )));
                }
                Some(Token::WhiteSpace(_)) => {
                    self.advance();
                    continue;
                }
                _ => return Ok((first_expression, token_info)),
            }
        }
    }

    fn parse_term(&mut self) -> Result<(Either, TokenInfo), Error> {
        let (mut first_expression, token_info) = self.parse_prod()?;
        loop {
            match self.peek().cloned() {
                Some(Token::Operator(
                    operator_token @ (OperatorToken::PLUS | OperatorToken::MINUS),
                    token_info,
                )) if first_expression.tipe() == &LanguageType::Number => {
                    self.advance();
                    let (second_expression, token_info) = self.parse_prod()?;
                    if first_expression.tipe() == second_expression.tipe() {
                        let expr = BinaryExpression::new(
                            first_expression,
                            second_expression,
                            operator_token,
                        );
                        first_expression = expr;
                        continue;
                    } else {
                        return Err(Error::TypeError(format!(
                            "Type Error: Type mismatched at line {}, column {}",
                            token_info.line_number, token_info.column_number
                        )));
                    }
                }
                Some(Token::Operator(
                    operator_token @ (OperatorToken::PLUS | OperatorToken::MINUS),
                    token_info,
                )) if first_expression.tipe() == &LanguageType::Boolean
                    || first_expression.tipe() == &LanguageType::Void =>
                {
                    return Err(Error::TypeError(format!(
                        "Type Error: Operator mismatched at line {}, column {}",
                        token_info.line_number, token_info.column_number
                    )));
                }
                Some(Token::WhiteSpace(_)) => {
                    self.advance();
                    continue;
                }
                _ => {
                    return Ok((first_expression, token_info));
                }
            }
        }
    }

    fn parse_prod(&mut self) -> Result<(Either, TokenInfo), Error> {
        let (mut first_expression, token_info) = self.parse_unary()?;
        loop {
            match self.peek().cloned() {
                Some(Token::Operator(
                    operator_token @ (OperatorToken::STAR | OperatorToken::DIVIDE),
                    token_info,
                )) if first_expression.tipe() == &LanguageType::Number => {
                    self.advance();
                    let (second_expression, token_info) = self.parse_unary()?;
                    if first_expression.tipe() == second_expression.tipe() {
                        let expr = BinaryExpression::new(
                            first_expression,
                            second_expression,
                            operator_token,
                        );
                        first_expression = expr;
                        continue;
                    } else {
                        return Err(Error::TypeError(format!(
                            "Type Error: Type mismatched at line {}, column {}",
                            token_info.line_number, token_info.column_number
                        )));
                    }
                }
                Some(Token::Operator(
                    operator_token @ (OperatorToken::STAR | OperatorToken::DIVIDE),
                    token_info,
                )) if first_expression.tipe() == &LanguageType::Boolean
                    || first_expression.tipe() == &LanguageType::Void =>
                {
                    return Err(Error::TypeError(format!(
                        "Type Error: Operator mismatched at line {}, column {}",
                        token_info.line_number, token_info.column_number
                    )));
                }

                Some(Token::WhiteSpace(_)) => {
                    self.advance();
                    continue;
                }
                _ => {
                    return Ok((first_expression, token_info));
                }
            }
        }
    }

    fn parse_unary(&mut self) -> Result<(Either, TokenInfo), Error> {
        match self.peek().cloned() {
            Some(Token::Operator(
                operator @ (OperatorToken::PLUS | OperatorToken::MINUS),
                token_info,
            )) => {
                self.advance();
                let (expr, token_info) = self.parse_unary()?;
                let tipe = expr.tipe().clone();
                let unary_expr = UnaryExpression::new(expr, operator);
                if tipe == LanguageType::Number {
                    Ok((unary_expr, token_info))
                } else {
                    Err(Error::TypeError(format!(
                        "Type Error: Type mismatched at line {}, column {}",
                        token_info.line_number, token_info.column_number
                    )))
                }
            }
            Some(Token::Operator(operator @ OperatorToken::LOGICAL_NOT, token_info)) => {
                self.advance();
                let (expr, token_info) = self.parse_unary()?;
                let tipe = expr.tipe().clone();
                let unary_expr = UnaryExpression::new(expr, operator);
                if tipe == LanguageType::Boolean {
                    Ok((unary_expr, token_info))
                } else {
                    Err(Error::TypeError(format!(
                        "Type Error: Type mismatched at line {}, column {}",
                        token_info.line_number, token_info.column_number
                    )))
                }
            }
            Some(Token::WhiteSpace(_)) => {
                self.advance();
                self.parse_unary()
            }
            _ => self.parse_bracket(),
        }
    }

    fn parse_bracket(&mut self) -> Result<(Either, TokenInfo), Error> {
        match self.peek().cloned() {
            Some(Token::Symbol(SymbolToken::SmallBracketOpen, token_info)) => {
                self.advance();
                let (val, token_info) = self.parse_expression()?;
                match self.peek().cloned() {
                    Some(Token::Symbol(SymbolToken::SmallBracketClose, _)) => {
                        self.advance();
                        Ok((val, token_info))
                    }
                    Some(Token::WhiteSpace(_)) => {
                        self.advance();
                        self.parse_bracket()
                    }
                    Some(token) => Err(Error::ParseError(format!(
                        "Parsing Error: No closing bracket found at line {}, column {}",
                        token.get_token_info().line_number,
                        token.get_token_info().column_number
                    ))),
                    None => Err(Error::ParseError(format!(
                        "Parsing Error: None value encountered"
                    ))),
                }
            }
            _ => self.parse_block_expression(),
        }
    }

    fn parse_block_expression(&mut self) -> Result<(Either, TokenInfo), Error> {
        match self.peek().cloned() {
            Some(Token::Symbol(SymbolToken::CurlyBracketOpen, token_info)) => {
                // println!("debug curly: {:?}", "here");
                self.increment_block_position();

                self.advance();
                self.consume_optional_whitespace();
                let mut statements: Vec<Either> = Vec::new();
                let mut last_statement_semicolon = false;
                loop {
                    match self.peek().cloned() {
                        Some(Token::Symbol(SymbolToken::CurlyBracketClose, _)) => {
                            self.decrement_block_position();

                            self.advance();
                            self.consume_optional_semicolon();
                            match statements.last() {
                                Some(last_statement) if !last_statement_semicolon => {
                                    let tipe = last_statement.tipe().clone();
                                    return Ok((
                                        Either::BlockStatement(statements, tipe),
                                        token_info,
                                    ));
                                }
                                Some(_) => {
                                    return Ok((
                                        Either::BlockStatement(statements, LanguageType::Void),
                                        token_info,
                                    ));
                                }
                                None => {
                                    return Ok((
                                        Either::BlockStatement(statements, LanguageType::Void),
                                        token_info,
                                    ));
                                }
                            }
                        }
                        Some(_) => {
                            let statement = self.parse_next_statement()?;
                            last_statement_semicolon = self.consume_optional_semicolon();
                            self.consume_optional_whitespace();
                            // println!("debug curly: {:?} {:?}", statement, self.peek());
                            statements.push(statement);
                        }
                        None => {
                            return Err(Error::TypeError(format!("None encountered")));
                        }
                    }
                }
            }
            _ => self.parse_identifier_expression(),
        }
    }

    fn parse_identifier_expression(&mut self) -> Result<(Either, TokenInfo), Error> {
        match self.peek().cloned() {
            Some(Token::Identifier(var_name, token_info)) => {
                // println!("debug in parse_literal {:?}", var_name);
                self.advance();
                let var_name = var_name.iter().collect::<String>();
                match self.get_reference_to_identifier(&var_name) {
                    Some(identifier) => Ok((identifier.clone_to_ast(), token_info)),
                    None => Err(Error::IdentifierError(format!(
                "Identifier Error: No variable named '{}' found in scope at line {}, column {}",
                var_name,
                token_info.line_number,
                token_info.column_number,
            ))),
                }
            }
            _ => self.parse_number_literal(),
        }
    }
    fn parse_number_literal(&mut self) -> Result<(Either, TokenInfo), Error> {
        match self.peek().cloned() {
            Some(Token::Digit(first_digit, digit_token)) => {
                let mut digits = vec![first_digit];
                let mut num_of_dot = 0;
                let mut token_info = digit_token;
                self.advance();
                loop {
                    match self.peek().cloned() {
                        Some(Token::Symbol(SymbolToken::Dot, _)) if num_of_dot < 1 => {
                            self.advance();
                            num_of_dot += 1;
                            digits.push(&'.');
                        }
                        Some(Token::Symbol(SymbolToken::Dot, token_info_symbol))
                            if num_of_dot >= 1 =>
                        {
                            return Err(Error::SyntaxError(format!(
                                "Syntax Error: Extra decimal(.) encountered at line {}, column {}",
                                token_info_symbol.line_number, token_info_symbol.column_number,
                            )));
                        }
                        Some(Token::Digit(val, digit_token)) => {
                            self.advance();
                            token_info = digit_token;
                            digits.push(val);
                        }

                        _ => {
                            break;
                        }
                    }
                }
                Ok((
                    Either::Number(
                        digits
                            .into_iter()
                            .collect::<String>()
                            .parse::<f64>()
                            .unwrap(),
                    ),
                    digit_token,
                ))
            }
            _ => self.parse_literal(),
        }
    }

    fn parse_literal(&mut self) -> Result<(Either, TokenInfo), Error> {
        match self.peek().cloned() {
            Some(Token::Keyword(KeywordToken::True, token_info)) => {
                self.advance();
                Ok((Either::Bool(true), token_info))
            }
            Some(Token::Keyword(KeywordToken::False, token_info)) => {
                self.advance();
                Ok((Either::Bool(false), token_info))
            }

            Some(token) => Err(Error::SyntaxError(format!(
                "Syntax Error: Value other than literal (Number, Boolean...) '{}' encountered at line {}, column {}",
                token.value(),
                token.get_token_info().line_number,
                token.get_token_info().column_number,
            ))),

            None => unreachable!(),
        }
    }
}
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
enum LanguageType {
    Number,
    Boolean,
    Void,

    PartiallyTyped,
    Untyped,
}
#[derive(Debug)]
enum Either {
    WhileStatement(Box<Either>, Box<Either>),
    LetStatement(Box<LetStatement>),
    AssignmentStatement(Box<AssignmentStatement>),

    IfElseStatement(Box<Either>, Box<Either>, Box<Either>, LanguageType),
    BinaryExpression(Box<BinaryExpression>),
    UnaryExpression(Box<UnaryExpression>),
    BlockStatement(Vec<Either>, LanguageType),
    Identifier(Identifier),
    Number(f64),
    Bool(bool),

    Placeholder(LanguageType),

    HaltStatement,
}

impl Either {
    fn tipe(&self) -> &LanguageType {
        match self {
            Self::Number(_) => &LanguageType::Number,
            Self::Bool(_) => &LanguageType::Boolean,
            Self::BinaryExpression(boxed_binary_expression) => boxed_binary_expression.tipe(),
            Self::UnaryExpression(boxed_unary_expression) => boxed_unary_expression.tipe(),
            Self::LetStatement(boxed_let_statement) => boxed_let_statement.tipe(),
            Self::AssignmentStatement(boxed_assignment_statement) => {
                boxed_assignment_statement.tipe()
            }
            Self::Identifier(identifier) => identifier.tipe(),
            Self::BlockStatement(_, tipe) => tipe,
            Self::WhileStatement(_, _) => &LanguageType::Void,
            Self::IfElseStatement(_, _, _, tipe) => tipe,

            Self::Placeholder(tipe) => tipe,
            Self::HaltStatement => &LanguageType::Void,
        }
    }
}

#[derive(Debug)]
enum Error {
    ParseError(String),
    TypeError(String),
    IdentifierError(String),
    SyntaxError(String),
}

#[derive(Debug)]
struct BinaryExpression {
    left: Either,
    right: Either,
    operator: OperatorToken,
}
impl BinaryExpression {
    fn new(left: Either, right: Either, operator: OperatorToken) -> Either {
        Either::BinaryExpression(Box::new(Self {
            left: left,
            right: right,
            operator: operator,
        }))
    }
    fn tipe(&self) -> &LanguageType {
        self.operator.tipe()
    }
}
#[derive(Debug)]
struct UnaryExpression {
    val: Either,
    operator: OperatorToken,
}
impl UnaryExpression {
    fn new(val: Either, operator: OperatorToken) -> Either {
        Either::UnaryExpression(Box::new(Self {
            val: val,
            operator: operator,
        }))
    }
    fn tipe(&self) -> &LanguageType {
        self.operator.tipe()
    }
}

#[derive(Debug)]
struct AssignmentStatement {
    lvalue: String,
    rvalue: Either,
}

impl AssignmentStatement {
    fn new(lvalue: String, rvalue: Either) -> Either {
        Either::AssignmentStatement(Box::new(Self {
            lvalue: lvalue,
            rvalue: rvalue,
        }))
    }
    fn tipe(&self) -> &LanguageType {
        // self.rvalue.tipe()
        &LanguageType::Void
    }
}
#[derive(Debug)]
struct LetStatement {
    lvalue: String,
    rvalue: Either,
}

impl LetStatement {
    fn new(lvalue: String, rvalue: Either) -> Either {
        Either::LetStatement(Box::new(Self {
            lvalue: lvalue,
            rvalue: rvalue,
        }))
    }
    fn tipe(&self) -> &LanguageType {
        // self.rvalue.tipe()
        &LanguageType::Void
    }
}

#[derive(Debug)]
struct Identifier {
    name: String,
    block_position: usize,
    tipe: LanguageType,
}

impl Identifier {
    fn new(name: &str, block_position: usize, tipe: LanguageType) -> Self {
        Self {
            name: name.to_string(),
            block_position: block_position,
            tipe: tipe,
        }
    }
    fn new_ast(name: String, block_position: usize, tipe: LanguageType) -> Either {
        Either::Identifier(Self {
            name: name,
            block_position: block_position,
            tipe: tipe,
        })
    }
    fn tipe(&self) -> &LanguageType {
        &self.tipe
    }
    fn clone(&self) -> Identifier {
        Self::new(&self.name, self.block_position, self.tipe)
    }
    fn clone_to_ast(&self) -> Either {
        Self::new_ast(self.name.to_string(), self.block_position, self.tipe)
    }
}

///////////////////////////////////////////////////////// Interpreter Code /////////////////////////////////////////////////
#[derive(Debug, Clone, Copy)]
enum InternalDataStucture {
    Number(f64),
    Bool(bool),
    Void,
}

impl InternalDataStucture {
    fn __match_binary_operation__(left: Self, right: Self, operator: OperatorToken) -> Self {
        match (left, right) {
            (Self::Number(l), Self::Number(r)) => match operator {
                OperatorToken::PLUS => Self::Number(l + r),
                OperatorToken::MINUS => Self::Number(l - r),
                OperatorToken::STAR => Self::Number(l * r),
                OperatorToken::DIVIDE => Self::Number(l / r),

                OperatorToken::LESS_THAN => Self::Bool(l < r),
                OperatorToken::GREATER_THAN => Self::Bool(l > r),
                OperatorToken::EQUAL_TO => Self::Bool(l == r),

                _ => {
                    println!("{:?}, {:?}", left, right);
                    unreachable!();
                }
            },

            (Self::Bool(l), Self::Bool(r)) => match operator {
                OperatorToken::LOGICAL_AND => Self::Bool(l && r),
                OperatorToken::LOGICAL_OR => Self::Bool(l || r),
                OperatorToken::EQUAL_TO => Self::Bool(l == r),
                _ => {
                    println!("{:?}, {:?}", left, right);
                    unreachable!()
                }
            },

            _ => {
                println!("{:?}, {:?}", left, right);
                unreachable!()
            }
        }
    }
    fn __match_unary_operation__(val: Self, operator: OperatorToken) -> Self {
        match val {
            Self::Number(v) => match operator {
                OperatorToken::PLUS => Self::Number(v),
                OperatorToken::MINUS => Self::Number(-v),
                _ => unreachable!(),
            },
            Self::Bool(b) => match operator {
                OperatorToken::LOGICAL_NOT => Self::Bool(!b),
                _ => unreachable!(),
            },

            _ => unreachable!(),
        }
    }
}
#[derive(Debug)]
struct Environment {
    container: Vec<HashMap<String, InternalDataStucture>>,
}

impl Environment {
    fn new() -> Self {
        Self {
            container: vec![HashMap::new()],
        }
    }
    fn insert_new(&mut self, lvalue: String, rvalue: InternalDataStucture) {
        self.container.last_mut().unwrap().insert(lvalue, rvalue);
    }
    fn insert(&mut self, lvalue: String, rvalue: InternalDataStucture) {
        let mut location = 0;
        for (i, child_container) in self.container.iter().enumerate().rev() {
            if child_container.contains_key(&lvalue) {
                location = i;
            }
        }
        self.container[location].insert(lvalue, rvalue);
    }
    fn get(&self, name: &str) -> Option<&InternalDataStucture> {
        for container in self.container.iter().rev() {
            if container.contains_key(name) {
                return container.get(name);
            }
        }
        None
    }
    fn create_child(&mut self) {
        self.container.push(HashMap::new());
    }
    fn pop_child(&mut self) {
        self.container.pop();
    }
}
struct Interpreter {
    environment: Environment,
    previous_identifiers: Option<Vec<Identifier>>,
}

impl Interpreter {
    fn new() -> Self {
        Interpreter {
            environment: Environment::new(),
            previous_identifiers: None,
        }
    }

    fn calculate_statement(&mut self, statement: &Either) -> InternalDataStucture {
        match statement {
            Either::Number(val) => InternalDataStucture::Number(*val),

            Either::Bool(val) => InternalDataStucture::Bool(*val),

            Either::BinaryExpression(boxed_expression) => match &**boxed_expression {
                BinaryExpression {
                    left,
                    right,
                    operator,
                } => InternalDataStucture::__match_binary_operation__(
                    self.calculate_statement(&left),
                    self.calculate_statement(&right),
                    *operator,
                ),
            },
            Either::UnaryExpression(boxed_expression) => match &**boxed_expression {
                UnaryExpression { val, operator } => {
                    InternalDataStucture::__match_unary_operation__(
                        self.calculate_statement(&val),
                        *operator,
                    )
                }
            },
            Either::LetStatement(boxed_let_statement) => match &**boxed_let_statement {
                LetStatement { lvalue, rvalue } => {
                    let return_val = self.calculate_statement(&rvalue);
                    self.environment.insert_new(lvalue.clone(), return_val);
                    InternalDataStucture::Void
                }
            },
            Either::AssignmentStatement(boxed_assignment_statement) => {
                match &**boxed_assignment_statement {
                    AssignmentStatement { lvalue, rvalue } => {
                        let return_val = self.calculate_statement(&rvalue);
                        self.environment.insert(lvalue.clone(), return_val);
                        InternalDataStucture::Void
                    }
                }
            }
            Either::Identifier(identifier) => {
                self.environment.get(&identifier.name).unwrap().clone()
            }

            Either::BlockStatement(statements, tipe) => {
                let mut result = InternalDataStucture::Void;
                self.environment.create_child();
                for statement in statements {
                    result = self.calculate_statement(statement);
                }
                self.environment.pop_child();
                match tipe {
                    LanguageType::Void => InternalDataStucture::Void,
                    _ => result,
                }
            }

            Either::WhileStatement(condition, block_statement) => {
                let mut result = InternalDataStucture::Void;
                loop {
                    match self.calculate_statement(condition) {
                        InternalDataStucture::Bool(b) if b == true => {
                            result = self.calculate_statement(block_statement);
                        }
                        InternalDataStucture::Bool(b) if b == false => break,
                        _ => unreachable!(),
                    }
                }
                result
            }
            Either::IfElseStatement(condition, if_block_statement, else_block_statement, _) => {
                let result;
                match self.calculate_statement(condition) {
                    InternalDataStucture::Bool(b) if b == true => {
                        result = self.calculate_statement(if_block_statement);
                    }
                    InternalDataStucture::Bool(b) if b == false => {
                        result = self.calculate_statement(else_block_statement);
                    }
                    _ => unreachable!(),
                }
                result
            }
            Either::Placeholder(tipe) => InternalDataStucture::Void,
            Either::HaltStatement => InternalDataStucture::Void,
        }
    }

    fn tokenize_and_parse(
        &mut self,
        input: &str,
        previous_identifiers: Option<Vec<Identifier>>,
    ) -> Result<(Vec<Either>, Option<Vec<Identifier>>), Error> {
        let input_chars = input.chars().collect::<Vec<_>>();

        let mut tokenizer = Tokenizer::new(&input_chars);
        let tokens = tokenizer.tokenize();

        match tokens {
            Ok(token_list) => {
                let mut parser = Parser::new(&token_list);
                parser.set_previous_identifiers_list(previous_identifiers);

                match parser.parse() {
                    Ok(parsed_statements) => Ok((parsed_statements, parser.get_identifiers_list())),
                    Err(err) => {
                        self.set_previous_identifiers(parser.get_identifiers_list());
                        Err(err)
                    }
                }
            }
            Err(err) => {
                self.set_previous_identifiers(previous_identifiers);
                Err(Error::ParseError(
                    err.into_iter()
                        .map(|x| x + "\n")
                        .collect::<String>()
                        .trim_end()
                        .to_string(),
                ))
            }
        }
    }
    fn set_previous_identifiers(&mut self, previous_identifiers: Option<Vec<Identifier>>) {
        self.previous_identifiers = previous_identifiers;
    }
    fn interpret(&mut self, input: &str) {
        let previous_identifiers = self.previous_identifiers.take();
        match self.tokenize_and_parse(input, previous_identifiers) {
            Ok((parsed_statements, new_identifiers)) => {
                for statement in parsed_statements {
                    match self.calculate_statement(&statement) {
                        InternalDataStucture::Void => continue,
                        result => println!("{:?}", result),
                    }
                }
                self.set_previous_identifiers(new_identifiers);
            }
            Err(error) => {
                println!("{:?}", error);
            }
        }
        // println!("{:?}", self.environment);
    }

    fn interactive_prompt(&mut self) {
        'loop_start: loop {
            let mut line = String::new();
            print!(">>> ");

            let _ = std::io::stdout().flush();
            let _ = std::io::stdin().read_line(&mut line).unwrap();

            let line = line.trim_end();

            if line.len() == 0 {
                continue 'loop_start;
            }
            self.interpret(&line);
        }
    }
}
